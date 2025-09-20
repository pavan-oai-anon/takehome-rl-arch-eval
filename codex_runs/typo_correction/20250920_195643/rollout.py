from __future__ import annotations

from typing import Any

import art
import weave
from openai import AsyncOpenAI

from env import (
    SYSTEM_PROMPT,
    evaluate_response,
    episode_metadata,
    format_user_prompt,
    resolve_generation_config,
    sample_episode,
)


@weave.op
@art.retry()
async def rollout(model: art.Model, step: int, config: dict[str, Any]) -> art.Trajectory:
    """Gather a single review correction trajectory."""

    config = config or {}
    generation_config = resolve_generation_config(config)
    episode = sample_episode(step)
    trajectory = art.Trajectory(
        messages_and_choices=[{"role": "system", "content": SYSTEM_PROMPT}],
        metadata=episode_metadata(step, episode),
        reward=0.0,
    )

    user_prompt = format_user_prompt(episode)
    trajectory.messages_and_choices.append({"role": "user", "content": user_prompt})

    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )
    completion = await client.chat.completions.create(
        model=model.name,
        messages=trajectory.messages(),
        max_completion_tokens=generation_config["max_completion_tokens"],
        temperature=generation_config["temperature"],
        top_p=generation_config["top_p"],
        stream=False,
    )

    choice = completion.choices[0]
    content = choice.message.content or ""
    trajectory.messages_and_choices.append(choice)

    evaluation = evaluate_response(content, episode=episode)
    trajectory.reward = evaluation.reward
    trajectory.metadata["validation_error"] = evaluation.error_message or "none"
    trajectory.metadata["invalid_flag"] = 1 if evaluation.error_message else 0
    trajectory.metadata["finish_reason"] = choice.finish_reason or "unknown"

    trajectory.metrics["similarity"] = evaluation.similarity
    trajectory.metrics["exact_match"] = 1.0 if evaluation.exact_match else 0.0
    trajectory.metrics["invalid_solution"] = evaluation.invalid_penalty
    trajectory.metrics["response_length"] = float(len(content.strip()))
    trajectory.metrics["target_length"] = float(len(episode.clean_text))

    return trajectory
