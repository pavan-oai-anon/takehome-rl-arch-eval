from __future__ import annotations

import random
from typing import Any

import art
import weave
from openai import AsyncOpenAI

from env import (
    RANDOM_SEED,
    SYSTEM_PROMPT,
    TRAINING_CONFIG,
    build_user_prompt,
    sample_example,
    validate_agent_response,
)


@weave.op
@art.retry(exceptions=(RuntimeError, ValueError), max_attempts=3)
async def rollout(model: art.Model, step: int, config: dict[str, Any]) -> art.Trajectory:
    """Run a single ART rollout that validates JSON extraction."""

    rng = random.Random(RANDOM_SEED + step)
    example = sample_example(rng)

    trajectory = art.Trajectory(
        messages_and_choices=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(example)},
        ],
        metadata={
            "example_id": example.example_id,
            "model_name": model.name,
            "step": step,
            "validation_error": "pending",
            "raw_char_count": len(example.raw_text),
        },
        reward=0.0,
    )

    temperature = float(config.get("temperature", TRAINING_CONFIG["temperature"]))
    top_p = float(config.get("top_p", TRAINING_CONFIG["top_p"]))
    max_tokens = int(config.get("max_completion_tokens", TRAINING_CONFIG["max_completion_tokens"]))

    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )

    completion = await client.chat.completions.create(
        model=model.name,
        messages=trajectory.messages(),
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_tokens,
        stream=False,
    )

    choice = completion.choices[0]
    content = choice.message.content or ""
    trajectory.messages_and_choices.append(choice)

    result = validate_agent_response(content, example)
    trajectory.reward = result.reward
    trajectory.metadata["validation_error"] = result.error
    trajectory.metadata["finish_reason"] = choice.finish_reason or "none"

    trajectory.metrics["invalid_solution"] = 1.0 if result.invalid else 0.0
    trajectory.metrics["field_accuracy"] = result.field_accuracy
    trajectory.metrics["used_temperature"] = temperature
    trajectory.metrics["used_top_p"] = top_p
    trajectory.metrics["max_completion_tokens"] = float(max_tokens)

    if result.invalid:
        trajectory.metrics["reward_penalty"] = result.reward
    else:
        trajectory.metrics["reward_penalty"] = 0.0

    return trajectory


__all__ = ["rollout"]
