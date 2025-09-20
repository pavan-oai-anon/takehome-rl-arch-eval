from __future__ import annotations

from typing import Any

import art
import weave
from openai import AsyncOpenAI

from env import (
    SYSTEM_PROMPT,
    TRAINING_CONFIG,
    sample_episode,
    render_user_prompt,
    validate_response,
)


@weave.op()
@art.retry()
async def rollout(model: art.Model, step: int, config: dict[str, Any]) -> art.Trajectory:
    """Generate a single ART trajectory for the typo correction task."""

    attempt = int(config.get("attempt", 0))
    episode = sample_episode(step, attempt=attempt)

    trajectory = art.Trajectory(
        messages_and_choices=[{"role": "system", "content": SYSTEM_PROMPT}],
        metadata={
            "step": step,
            "review_id": episode.episode_id,
            "difficulty": episode.difficulty or "none",
            "attempt": attempt,
        },
        reward=0.0,
    )

    trajectory.messages_and_choices.append(
        {"role": "user", "content": render_user_prompt(episode)}
    )

    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )

    chat_completion = await client.chat.completions.create(
        model=model.name,
        messages=trajectory.messages(),
        max_completion_tokens=int(
            config.get("max_completion_tokens", TRAINING_CONFIG["max_completion_tokens"])
        ),
        temperature=float(config.get("temperature", TRAINING_CONFIG["temperature"])),
        top_p=float(config.get("top_p", TRAINING_CONFIG["top_p"])),
        stream=False,
    )

    choice = chat_completion.choices[0]
    content = choice.message.content or ""

    trajectory.messages_and_choices.append(
        {"role": "assistant", "content": content}
    )

    validation = validate_response(content, episode)

    trajectory.reward = validation.reward
    trajectory.metrics["exact_match"] = 1.0 if validation.exact_match else 0.0
    trajectory.metrics["invalid_solution"] = 1.0 if validation.invalid else 0.0
    trajectory.metrics["response_length"] = float(len(content.strip()))
    trajectory.metadata["validation_error"] = validation.error or ""

    return trajectory
