"""ART rollout for the typo-correction environment.

Implements a single-episode rollout that:
- surfaces one noisy review (plus a difficulty hint)
- asks the model to return the corrected review verbatim
- validates the output defensively and computes smooth rewards

Assumes a LocalBackend for inference/training. For GPU memory, you may
reuse the tuning values seen in the 2048 example if needed.

Dependencies (install if needed):
- uv pip install openai weave requests
"""
from __future__ import annotations

from typing import Any

import art
import requests
import weave
from openai import AsyncOpenAI

from env import (
    SYSTEM_PROMPT,
    TRAINING_CONFIG,
    difficulty_code,
    select_example,
    user_prompt,
    compute_metrics_and_reward,
)


@weave.op
@art.retry(exceptions=(requests.ReadTimeout,))
async def rollout(model: art.Model, step: int, config: dict[str, Any]) -> art.Trajectory:
    """Run one correction episode and return an ART trajectory.

    Args:
        model: ART model wrapper (inference endpoint and credentials)
        step: current global training step (used to pick an example)
        config: host-provided knobs (falls back to TRAINING_CONFIG)

    Returns:
        art.Trajectory with messages, scalar metadata, metrics, and reward.
    """

    # Resolve runtime config with sane fallbacks.
    max_tokens = int(config.get("max_completion_tokens", TRAINING_CONFIG["max_completion_tokens"]))
    temperature = float(config.get("temperature", TRAINING_CONFIG["temperature"]))
    top_p = float(config.get("top_p", TRAINING_CONFIG["top_p"]))

    # Sample one episode deterministically by step.
    ex = select_example(step)

    # Prepare initial trajectory with the system instruction only.
    trajectory = art.Trajectory(
        messages_and_choices=[{"role": "system", "content": SYSTEM_PROMPT}],
        metadata={
            "episode_id": ex.rid,
            "difficulty_label": ex.difficulty,
            "difficulty_code": difficulty_code(ex.difficulty),
            "step": int(step),
            "validation_error": "",  # filled after validation if any
            "notebook_id": "typo-correction",  # scalar string OK
            "noisy_len": len(ex.noisy),
            "target_len": len(ex.clean),
        },
        reward=0.0,
    )

    # Add the user prompt containing the noisy review and constraint reminder.
    trajectory.messages_and_choices.append({"role": "user", "content": user_prompt(ex)})

    # Query the model via its configured OpenAI-compatible endpoint.
    client = AsyncOpenAI(base_url=model.inference_base_url, api_key=model.inference_api_key)
    completion = await client.chat.completions.create(
        messages=trajectory.messages(),
        model=model.name,
        stream=False,
        max_completion_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    choice = completion.choices[0]
    content = choice.message.content or ""
    trajectory.messages_and_choices.append(choice)

    # Validate and score.
    reward, metrics, format_violation, err = compute_metrics_and_reward(content, ex.clean)
    trajectory.reward = reward
    for k, v in metrics.items():
        trajectory.metrics[k] = float(v)

    # Record scalar metadata (no lists/dicts) for aggregation.
    if format_violation or err:
        trajectory.metadata["validation_error"] = err or "format_violation"
    else:
        trajectory.metadata["validation_error"] = ""

    return trajectory


__all__ = ["rollout"]

