"""ART rollout logic for the *Exact Word-Count Story* environment.

This file is intentionally compact – it focuses solely on collecting a single
trajectory: sampling a scenario, querying the model, validating the response,
and attaching shaped rewards + metrics so ART can train adapters.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET  # noqa: F401 (kept for parity with 2048 example)
from typing import Any

import art
import requests
import weave
from openai import AsyncOpenAI

import env  # Local helpers


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


@weave.op
@art.retry(exceptions=(requests.ReadTimeout,))
async def rollout(
    model: art.Model,
    step: int,
    config: dict[str, Any],  # noqa: D401 – Config forwarded by caller.
) -> art.Trajectory:  # pragma: no cover – runtime function.
    """Collect one trajectory for a *story-word-count* scenario.

    Parameters
    ----------
    model:  The current *trainable* or *inference* model from ART.
    step:   Training step so far – forwarded by ART's generic loop.
    config: Unused free-form configuration dictionary (reserved for future).
    """

    # ---------------------------------------------------------------------
    # 1. Sample / build scenario & tracking containers.
    # ---------------------------------------------------------------------

    scenario = env.sample_scenario(step)

    # System instruction includes formatting constraints – used every turn so
    # the model cannot claim it never saw the requirement.
    system_prompt = (
        "You are a concise creative writer.  Respond with *plain text only* – "
        "no quotes, no bullet points, no numbering, and exactly the required "
        "word count.  Contractions count as one word.  Avoid filler words."  # noqa: E501
    )

    # "User" message describes the concrete task.
    user_prompt = (
        f"Write a short story of exactly {scenario.word_count} words about "
        f"{scenario.theme}."
    )

    trajectory = art.Trajectory(
        messages_and_choices=[{"role": "system", "content": system_prompt}],
        metadata={
            "step": step,
            "theme": scenario.theme,
            "target_word_count": scenario.word_count,
        },
        reward=0.0,
    )

    # Append user prompt.
    trajectory.messages_and_choices.append({"role": "user", "content": user_prompt})

    # ---------------------------------------------------------------------
    # 2. Query model.
    # ---------------------------------------------------------------------

    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )

    chat_completion = await client.chat.completions.create(
        messages=trajectory.messages(),
        model=model.name,
        max_completion_tokens=env.TRAINING_CONFIG["max_completion_tokens"],
        temperature=env.TRAINING_CONFIG["temperature"],
        top_p=env.TRAINING_CONFIG["top_p"],
        stream=False,
    )

    choice = chat_completion.choices[0]
    content = choice.message.content or ""
    trajectory.messages_and_choices.append(choice)

    # ---------------------------------------------------------------------
    # 3. Validate response & build rewards / metrics.
    # ---------------------------------------------------------------------

    actual_words = env.count_words(content)
    target_words = scenario.word_count
    diff = abs(actual_words - target_words)

    trajectory.metrics["word_count"] = actual_words
    trajectory.metrics["word_diff"] = diff

    # Binary flag so we can quickly slice datasets for troubleshooting.
    invalid_solution = float(diff != 0)
    trajectory.metrics["invalid_solution"] = invalid_solution

    # Reward shaping – dense and symmetric so PPO has gradient everywhere.
    trajectory.reward = env.word_count_reward(actual_words, target_words)

    # Record validation error in *metadata* (string scalar) for aggregation.
    if diff != 0:
        trajectory.metadata["error"] = (
            f"wrong word count: expected {target_words}, got {actual_words}"
        )

    return trajectory


# ---------------------------------------------------------------------------
# Convenience exposure so `training.py` can import without knowing internals.
# ---------------------------------------------------------------------------

__all__ = ["rollout"]

