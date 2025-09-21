"""ART rollout for exact word-count micro-stories.

Implements a single async rollout() function used by the host training loop.
Assumes a LocalBackend for inference/training. GPU selection and memory tuning
follow the same considerations as the 2048 example in ART (see that script for
PYTORCH_CUDA_ALLOC_CONF notes).
"""
from __future__ import annotations

from typing import Any

import art
import requests
import weave
from openai import AsyncOpenAI

from env import (
    RANDOM_SEED,
    TRAINING_CONFIG,
    build_system_prompt,
    build_user_prompt,
    choose_task,
    coverage_score,
    extract_keywords,
    few_shot_examples,
    reference_tasks,
    validate_story,
    word_count,
)


def _cfg(config: dict[str, Any], key: str) -> Any:
    """Get a config value with fallback to TRAINING_CONFIG."""

    if config is None:
        return TRAINING_CONFIG[key]
    return config.get(key, TRAINING_CONFIG[key])


@weave.op
@art.retry(exceptions=(requests.ReadTimeout,))
async def rollout(model: art.Model, step: int, config: dict[str, Any]) -> art.Trajectory:
    """Collect one trajectory for the word-count story environment.

    - Builds concise system + user prompts describing theme and exact length.
    - Adds 0–2 few-shot examples from the reference set.
    - Validates the assistant reply: word count, formatting.
    - Emits scalar metadata and numeric metrics.
    - Shapes a smooth reward emphasizing exact count and theme coverage.
    """

    # Deterministic RNG per step for examples and sampling variety.
    rnd = __import__("random").Random(RANDOM_SEED + step * 101)

    task = choose_task(step)
    sys_prompt = build_system_prompt()

    trajectory = art.Trajectory(
        messages_and_choices=[{"role": "system", "content": sys_prompt}],
        metadata={
            "step": int(step),
            "prompt_id": int(task.id),
            "theme": task.theme,
            "target_words": int(task.target_words),
            # Will be updated after validation; must be a scalar string.
            "validation_error": "",
            # Helpful scalar for aggregation.
            "has_examples": 0,
        },
        reward=0.0,
    )

    # Few-shot examples sampled from the reference set (scalars only in metadata).
    examples = few_shot_examples(rnd, k=2)
    if examples:
        trajectory.metadata["has_examples"] = 1
        for user_text, assistant_text in examples:
            trajectory.messages_and_choices.append({"role": "user", "content": user_text})
            trajectory.messages_and_choices.append({"role": "assistant", "content": assistant_text})

    # Actual user instruction for this rollout.
    user_prompt = build_user_prompt(task.theme, task.target_words)
    trajectory.messages_and_choices.append({"role": "user", "content": user_prompt})

    # Inference via OpenAI-compatible client using ART model endpoints.
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )

    chat_completion = await client.chat.completions.create(
        model=model.name,
        messages=trajectory.messages(),
        stream=False,
        max_completion_tokens=int(_cfg(config, "max_completion_tokens")),
        temperature=float(_cfg(config, "temperature")),
        top_p=float(_cfg(config, "top_p")),
    )

    choice = chat_completion.choices[0]
    content = choice.message.content
    if not isinstance(content, str):
        content = ""
    trajectory.messages_and_choices.append(choice)

    # Validation and scoring
    valid, err = validate_story(content, task.target_words)
    wc = word_count(content)
    target = int(task.target_words)
    deviation = abs(wc - target)
    exact = 1.0 if deviation == 0 else 0.0
    keys = extract_keywords(task.theme)
    cov = coverage_score(content, keys)
    # Smooth closeness reward in [0, 1]
    closeness = max(0.0, 1.0 - (deviation / max(1, target)))

    reward = 0.7 * closeness + 0.3 * cov
    if exact == 1.0:
        reward += 0.3  # exact-count bonus
    if not valid:
        reward -= 0.5  # formatting penalty
    # Clamp to a sane range
    reward = max(-1.0, min(1.5, reward))

    # Metrics: numbers only
    trajectory.metrics["word_count"] = float(wc)
    trajectory.metrics["target_words"] = float(target)
    trajectory.metrics["abs_deviation"] = float(deviation)
    trajectory.metrics["coverage"] = float(cov)
    trajectory.metrics["closeness"] = float(closeness)
    trajectory.metrics["exact_match"] = float(exact)
    trajectory.metrics["invalid_solution"] = 0.0 if valid else 1.0

    # Scalar metadata only
    trajectory.metadata["validation_error"] = err

    trajectory.reward = float(reward)
    return trajectory


# Optional utility for quick manual inspection (not used by the host).
def debug_available_tasks() -> str:
    """Return a compact string summary of reference tasks (for humans)."""

    items = reference_tasks()
    return ", ".join(f"#{t.id}:{t.theme}/{t.target_words}" for t in items)


# End of rollout.py

