"""ART rollout logic for the *typo-fix* task.

This file mirrors the style of the 2048 reference implementation while being
considerably simpler – each episode only contains **one** user turn. The agent
receives a typo-ridden product review and must respond **only** with the fully
corrected version.

Reward shaping uses Levenshtein-style similarity (via ``difflib``) to provide a
smooth signal between ``-1`` and ``+1`` so that partially-correct fixes still
contribute to learning.
"""

from __future__ import annotations

import requests
from openai import AsyncOpenAI
import art
import weave

from typing import Any

from env import (
    sample_episode,
    compute_reward,
    is_valid_output,
)


class ScenarioTypos(art.TypedDict):  # type: ignore[misc]
    """Minimal scenario container so that different training steps can be traced."""

    step: int


# ---------------------------------------------------------------------------
# Rollout definition.
# ---------------------------------------------------------------------------


@weave.op
@art.retry(exceptions=(requests.ReadTimeout,))
async def rollout(
    model: art.Model, step: int, config: dict[str, Any]  # noqa: D401 – ART signature
) -> art.Trajectory:
    """Generate a single trajectory for the *typo-fix* environment."""

    # ---------------------------------------------------------------------
    # Episode setup.
    # ---------------------------------------------------------------------
    noisy_review, correct_review, difficulty, review_id = sample_episode()

    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": (
                    "You are an expert proof-reader. Given a customer review that "
                    "may contain typos or grammar mistakes, respond with the fully "
                    "corrected review. Respond with *plain text only* – no extra "
                    "comments, no markdown, no explanations. The output must be "
                    "identical to the corrected version, including punctuation."
                ),
            }
        ],
        metadata={
            "review_id": float(review_id),  # scalar for aggregation
            "difficulty": difficulty,
            "step": float(step),
        },
        reward=0.0,
    )

    # Add user turn with the noisy text.
    trajectory.messages_and_choices.append({"role": "user", "content": noisy_review})

    # ------------------------------------------------------------------
    # Model inference.
    # ------------------------------------------------------------------
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )

    chat_completion = await client.chat.completions.create(
        model=model.name,
        messages=trajectory.messages(),
        max_completion_tokens=config.get("max_completion_tokens", 128),
        temperature=config.get("temperature", 0.7),
        top_p=config.get("top_p", 0.9),
        stream=False,
    )

    choice = chat_completion.choices[0]
    content = choice.message.content or ""
    trajectory.messages_and_choices.append(choice)

    # ------------------------------------------------------------------
    # Validation & reward.
    # ------------------------------------------------------------------
    is_valid = is_valid_output(content)
    trajectory.metrics["invalid_output"] = 0.0 if is_valid else 1.0

    if not is_valid:
        trajectory.reward = -1.0
        return trajectory

    similarity_reward = compute_reward(content, correct_review)

    # Emphasise perfect corrections while still learning from partial ones.
    trajectory.reward = similarity_reward

    # Additional metrics for analysis (all scalars).
    trajectory.metrics["similarity"] = similarity_reward

    return trajectory


# Re-export for ART's dynamic import pattern.
__all__ = ["rollout"]

