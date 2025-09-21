"""Rollout logic for typo correction tasks using ART and Weave."""
from __future__ import annotations

import random
import difflib
from typing import Any, Dict

import art
import weave
from openai import AsyncOpenAI

from env import SAMPLE_REVIEWS, RANDOM_SEED


@weave.op
@art.retry(exceptions=(ValueError,))
async def rollout(model: art.Model, step: int, config: Dict[str, Any]) -> art.Trajectory:
    """Generate one trajectory: correct typos in a product review."""
    # Seed for reproducible sample selection per step
    random.seed(RANDOM_SEED + step)
    sample = random.choice(SAMPLE_REVIEWS)
    review_id = int(sample["id"])
    noisy = sample["noisy"]
    clean = sample["clean"]
    difficulty = sample["difficulty"]

    # Initialize client for inference (LocalBackend assumed)
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )

    # Setup trajectory with system prompt
    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that corrects spelling and grammar mistakes"
                    " in product reviews. Only return the corrected review text verbatim."
                ),
            }
        ],
        metadata={
            "review_id": review_id,
            "step": step,
            "difficulty": difficulty,
        },
        reward=0.0,
    )

    # Add user message with noisy review
    trajectory.messages_and_choices.append(
        {"role": "user", "content": noisy}
    )

    # Request completion
    chat_completion = await client.chat.completions.create(
        model=model.name,
        messages=trajectory.messages(),
        max_tokens=config.get("max_completion_tokens"),
        temperature=config.get("temperature"),
        top_p=config.get("top_p"),
        stream=False,
    )
    choice = chat_completion.choices[0]
    content = choice.message.content.strip()
    trajectory.messages_and_choices.append(choice)

    # Defensive validation
    if not content:
        trajectory.metadata["invalid_solution"] = 1.0
        trajectory.metrics["invalid_solution"] = 1.0
        trajectory.reward = 0.0
        return trajectory

    # Compute similarity to ground truth
    similarity = difflib.SequenceMatcher(None, clean, content).ratio()
    # Record metrics and metadata as scalars
    trajectory.metrics["similarity"] = similarity
    trajectory.metrics["success"] = 1.0 if similarity == 1.0 else 0.0
    trajectory.metadata["invalid_solution"] = 0.0 if similarity == 1.0 else 1.0

    # Smooth reward based on similarity
    trajectory.reward = similarity
    return trajectory
