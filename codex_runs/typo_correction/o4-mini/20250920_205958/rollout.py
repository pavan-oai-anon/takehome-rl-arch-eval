"""
Rollout logic for the typo correction task using ART.
"""
from __future__ import annotations

import difflib
import requests

import art
import weave
from openai import AsyncOpenAI

from env import sample_review, RANDOM_SEED

# Assumes LocalBackend defaults; adjust GPU memory tuning if needed.

@weave.op
@art.retry(exceptions=(requests.ReadTimeout,))
async def rollout(
    model: art.Model,
    step: int,
    config: dict[str, object]
) -> art.Trajectory:
    """
    Execute one episode: provide a noisy review to the model and collect correction.

    Reward is the similarity ratio between original and corrected text.
    """
    # Initialize client for inference
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )

    # Sample a review based on step
    review = sample_review(step)
    original = review["text"]

    # Build initial trajectory
    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": (
                    "You are an expert copy editor. Correct the given product review. "
                    "Return only the corrected review verbatim."
                ),
            },
            {"role": "user", "content": original},
        ],
        metadata={
            "review_id": int(review["id"]),
            "difficulty": review["difficulty"],
            "step": step,
        },
        reward=0.0,
    )

    # Request model completion
    response = await client.chat.completions.create(
        model=model.name,
        messages=trajectory.messages(),
        max_completion_tokens=config["max_completion_tokens"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        stream=False,
    )
    choice = response.choices[0]
    content = choice.message.content.strip()
    trajectory.messages_and_choices.append(choice)

    # Defensive validation
    if not content:
        trajectory.metadata["validation_error"] = 1
        trajectory.reward = -1.0
        trajectory.metrics["invalid_solution"] = 1.0
        return trajectory

    # Smooth reward: similarity ratio
    similarity = difflib.SequenceMatcher(None, original, content).ratio()
    trajectory.metrics["similarity"] = similarity
    trajectory.reward = similarity

    return trajectory
