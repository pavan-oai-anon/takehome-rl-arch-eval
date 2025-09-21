"""
ART rollout logic for short-story word-count RL task.
Note: GPU memory tuning and LocalBackend setup happen in host training config.
"""
from __future__ import annotations
import re
from typing import Any

import art
import weave
from openai import AsyncOpenAI

from env import RANDOM_SEED, TRAINING_CONFIG, get_example, init_seed, STORY_EXAMPLES

@weave.op
@art.retry(exceptions=(Exception,))
async def rollout(model: art.Model, step: int, config: dict[str, Any]) -> art.Trajectory:
    """
    Generate a short story with exact word count and compute reward.
    """
    # Initialize randomness
    init_seed()

    # Select scenario based on step
    example = get_example(step)
    theme = example["theme"]
    target = example["word_count"]

    # Prepare OpenAI client
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )

    # Build initial trajectory with few-shot examples
    messages: list[dict[str, Any]] = []
    # Few-shot: reference prompts and solutions
    for ex in STORY_EXAMPLES:
        messages.append({"role": "user", "content": f"Theme: {ex['theme']}, exactly {ex['word_count']} words."})
        messages.append({"role": "assistant", "content": ex['reference']})

    # Actual scenario prompt
    prompt = f"Theme: {theme}, exactly {target} words. Write a plain-text story matching the count."
    messages.append({"role": "user", "content": prompt})

    # Initialize trajectory
    trajectory = art.Trajectory(
        messages_and_choices=[*messages],
        metadata={
            "step": step,
            "theme": theme,
            "target_word_count": target,
        },
        reward=0.0,
    )

    # Request completion (uses max_completion_tokens from config)
    response = await client.chat.completions.create(
        model=model.name,
        messages=trajectory.messages(),
        max_completion_tokens=config.get("max_completion_tokens"),
        temperature=config.get("temperature"),
        top_p=config.get("top_p"),
        stream=False,
    )
    choice = response.choices[0]
    content = choice.message.content
    # Append assistant choice
    trajectory.messages_and_choices.append(choice)

    # Validate output type
    if not isinstance(content, str):
        trajectory.metadata["error_message"] = "Non-string output"
        trajectory.metrics["invalid_solution"] = 1.0
        trajectory.reward = 0.0
        return trajectory

    # Count words (split by whitespace)
    words = re.findall(r"\S+", content)
    actual = len(words)
    diff = abs(actual - target)
    # Record metrics
    trajectory.metrics["actual_word_count"] = actual
    trajectory.metrics["word_count_diff"] = diff
    if diff != 0:
        trajectory.metadata["error_message"] = f"Expected {target}, got {actual}"  # record deviation

    # Reward: smooth penalty proportional to count deviation
    reward = max(0.0, 1.0 - (diff / target))
    trajectory.reward = reward
    return trajectory
