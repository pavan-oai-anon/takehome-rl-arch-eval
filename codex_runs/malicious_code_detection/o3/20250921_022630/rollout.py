"""ART rollout implementation for the malicious-code classification task."""

from __future__ import annotations

import json
from typing import Any

import art
import weave
import requests
from openai import AsyncOpenAI

import env


@weave.op
@art.retry(exceptions=(requests.ReadTimeout,))
async def rollout(
    model: art.Model,  # noqa: D401 – type from ART
    step: int,
    config: dict[str, Any],
) -> art.Trajectory:  # noqa: D401 – keep signature explicit
    """Single-episode rollout.

    The function signature matches OpenPipe ART expectations so the generic
    `training.py` harness can import and invoke it directly.
    """

    # ---------------------------------------------------------------------
    # 1. Sample environment state
    # ---------------------------------------------------------------------
    snippet = env.sample_snippet()

    # ---------------------------------------------------------------------
    # 2. Build initial trajectory with system message
    # ---------------------------------------------------------------------
    trajectory = art.Trajectory(
        messages_and_choices=[
            {"role": "system", "content": env.build_system_prompt()}
        ],
        metadata={
            "snippet_id": snippet.id,
            "step": step,
        },  # scalar values only
        reward=0.0,
    )

    # Add user prompt containing the snippet itself.
    trajectory.messages_and_choices.append(
        {
            "role": "user",
            "content": env.build_user_prompt(snippet),
        }
    )

    # ---------------------------------------------------------------------
    # 3. Model inference
    # ---------------------------------------------------------------------
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )

    chat_completion = await client.chat.completions.create(
        model=model.name,
        messages=trajectory.messages(),
        max_completion_tokens=config.get("max_completion_tokens", env.TRAINING_CONFIG["max_completion_tokens"]),
        temperature=config.get("temperature", env.TRAINING_CONFIG["temperature"]),
        top_p=config.get("top_p", env.TRAINING_CONFIG["top_p"]),
        stream=False,
    )

    choice = chat_completion.choices[0]
    trajectory.messages_and_choices.append(choice)

    # ---------------------------------------------------------------------
    # 4. Validation & reward calculation
    # ---------------------------------------------------------------------
    assistant_reply = choice.message.content
    assert isinstance(assistant_reply, str)

    validation_reward, metrics = env.validate_agent_response(assistant_reply)

    is_valid = metrics.get("invalid_json", 0.0) == 0.0

    if is_valid:
        # Safe to parse again – already validated.
        parsed = json.loads(assistant_reply)
        prediction = bool(parsed["is_malicious"])  # type: ignore[arg-type]
        is_correct = prediction == snippet.is_malicious
    else:
        is_correct = False

    reward = env.compute_reward(is_correct, is_valid)

    # ---------------------------------------------------------------------
    # 5. Populate trajectory metrics and reward
    # ---------------------------------------------------------------------
    trajectory.reward = reward

    # Store scalar numeric metrics only.
    trajectory.metrics["correct"] = 1.0 if is_correct else 0.0
    trajectory.metrics["invalid_json"] = metrics.get("invalid_json", 0.0)
    trajectory.metrics["label_malicious"] = 1.0 if snippet.is_malicious else 0.0

    return trajectory

