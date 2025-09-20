"""ART rollout logic for the JSON-extraction task.

The function signature follows the host framework requirements::

    async def rollout(model: art.Model, step: int, config: dict[str, Any])

The reward signal is dense: every correctly extracted field earns the agent
`1/len(SCHEMA)` and a perfect extraction grants a *bonus* for faster
convergence.  Invalid JSON or schema violations result in negative reward so
that the policy quickly learns to emit syntactically valid responses.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, Tuple

import art
import weave
from openai import AsyncOpenAI

import env

# ---------------------------------------------------------------------------
# Decoders & validation helpers
# ---------------------------------------------------------------------------


def _safe_json_loads(raw: str) -> Tuple[Dict[str, Any] | None, bool]:
    """Attempt to parse *raw* into JSON, stripping code-fence wrappers etc.

    Returns (parsed, had_error).
    """

    # Remove common ```json fences just in case.
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.IGNORECASE)

    try:
        return json.loads(cleaned), False
    except json.JSONDecodeError:
        return None, True


# ---------------------------------------------------------------------------
# Main rollout op
# ---------------------------------------------------------------------------


@weave.op
@art.retry(exceptions=(RuntimeError,))  # openai errors propagate as RuntimeError
async def rollout(model: art.Model, step: int, config: dict[str, Any]) -> art.Trajectory:  # noqa: D401,E501
    """Single RL episode – extract the schema from a random sample text."""

    # Deterministic sampling to ensure reproducibility across workers.
    random.seed(env.RANDOM_SEED + step)
    sample = env.sample_for_step(step)

    # System/user messages
    sys_prompt = (
        "You are an information extraction assistant. Given free-form text, "
        "output ONLY a compact JSON object matching this schema (no code "
        "fences, no additional keys):\n"
        f"Schema: {json.dumps(env.SCHEMA)}"
    )

    user_content = (
        "Extract the information according to the schema above. Text follows:\n\n"
        f"{sample.text}"
    )

    trajectory = art.Trajectory(
        messages_and_choices=[{"role": "system", "content": sys_prompt}],
        metadata={
            "sample_id": sample.id,  # scalar string per requirement
            "step": step,
        },
        reward=0.0,
    )

    # Append user message
    trajectory.messages_and_choices.append({"role": "user", "content": user_content})

    # ---- Inference ----
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )

    chat_response = await client.chat.completions.create(
        messages=trajectory.messages(),
        model=model.name,
        max_completion_tokens=config.get("max_completion_tokens", env.TRAINING_CONFIG["max_completion_tokens"]),
        temperature=config.get("temperature", env.TRAINING_CONFIG["temperature"]),
        top_p=config.get("top_p", env.TRAINING_CONFIG["top_p"]),
        stream=False,
    )

    choice = chat_response.choices[0]
    assistant_content = choice.message.content or ""
    trajectory.messages_and_choices.append(choice)

    # ---- Validation & reward shaping ----
    parsed_json, json_error = _safe_json_loads(assistant_content)

    invalid_json = 1.0 if json_error else 0.0
    trajectory.metrics["invalid_json"] = invalid_json

    if json_error or not isinstance(parsed_json, dict):
        trajectory.reward = -1.0
        return trajectory

    reward, field_metrics = env.grade_extraction(parsed_json, sample.ground_truth)

    # Record metrics as scalars.
    for key, value in field_metrics.items():
        trajectory.metrics[key] = value

    trajectory.reward = reward
    return trajectory
