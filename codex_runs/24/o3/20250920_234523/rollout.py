"""ART rollout logic for the *24 Game* environment.

The rollout is *single-turn*: we generate one puzzle, ask the model for an
answer, validate the XML payload, then compute a shaped reward using helpers
from :pymod:`env`.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import art
import requests
import weave
from openai import AsyncOpenAI

import env as _env

# ---------------------------------------------------------------------------
# Rollout implementation ------------------------------------------------------
# ---------------------------------------------------------------------------


@weave.op
@art.retry(exceptions=(requests.ReadTimeout,))
async def rollout(model: art.Model, step: int, config: dict[str, Any]) -> art.Trajectory:  # noqa: D401,E501
    """Generate one trajectory for the *24 Game* task.

    Parameters
    ----------
    model
        The `art.Model` providing inference.
    step
        Current RL step, forwarded from trainer.
    config
        Current training configuration.  **Unused** here but included for API
        compatibility.
    """

    # ---------------------------------------------------------------------
    # 1. Prepare puzzle & trajectory skeleton
    # ---------------------------------------------------------------------

    digits = _env.generate_puzzle()
    puzzle_str = _env.puzzle_to_string(digits)

    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": (
                    "You are playing the arithmetic game 24.  Your task is to take "
                    "four digits provided by the user and combine *each digit "
                    "exactly once* with the operators +, -, *, / and parentheses "
                    "so that the expression evaluates to exactly 24.\n\n"
                    "Rules:\n"
                    "1. Use every digit once and only once.\n"
                    "2. No other numbers/constants are allowed.\n"
                    "3. Respond *only* with XML of the form "
                    "<solution>(DIGIT OP DIGIT ...)</solution>."
                ),
            }
        ],
        metadata={
            "puzzle": puzzle_str,  # scalar str – allowed by ART
            "step": step,
        },
        reward=0.0,  # will be updated after validation
    )

    # ---------------------------------------------------------------------
    # 2. Invoke model
    # ---------------------------------------------------------------------

    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )

    trajectory.messages_and_choices.append({"role": "user", "content": puzzle_str})

    chat_completion = await client.chat.completions.create(
        model=model.name,
        messages=trajectory.messages(),
        max_completion_tokens=int(_env.TRAINING_CONFIG["max_completion_tokens"]),
        stream=False,
        temperature=float(_env.TRAINING_CONFIG["temperature"]),
        top_p=float(_env.TRAINING_CONFIG["top_p"]),
    )

    choice = chat_completion.choices[0]
    assert isinstance(choice.message.content, str)
    trajectory.messages_and_choices.append(choice)

    # ---------------------------------------------------------------------
    # 3. Parse & validate solution
    # ---------------------------------------------------------------------

    try:
        solution_xml = ET.fromstring(choice.message.content)
        if solution_xml.tag != "solution":  # enforce correct tag
            raise ValueError("root tag must be <solution>")
        expression = solution_xml.text or ""
    except ET.ParseError:
        expression = ""
        valid = False
        value = 0.0
        error = "invalid XML"
    else:
        valid, value, error = _env.validate_solution(expression, digits)

    # ---------------------------------------------------------------------
    # 4. Compute reward & record metrics/metadata
    # ---------------------------------------------------------------------

    reward = _env.reward_from_value(value, valid)

    trajectory.reward = reward
    trajectory.metrics["value"] = value
    trajectory.metrics["distance"] = abs(_env.TARGET - value)
    trajectory.metrics["valid_solution"] = 1.0 if valid else 0.0

    # Store validation error as string scalar if present so we can aggregate.
    if error:
        trajectory.metadata["error"] = error  # type: ignore[index]

    return trajectory


# ---------------------------------------------------------------------------
# End of file ----------------------------------------------------------------
# ---------------------------------------------------------------------------

