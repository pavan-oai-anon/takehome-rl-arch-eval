"""Rollout logic for ART typo correction task."""
from typing import Any
from env import TRAINING_CONFIG, prepare_review
import art
import weave


@weave.op
@art.retry(exceptions=(Exception,))
async def rollout(model: art.Model, step: int, config: dict[str, Any]) -> art.Trajectory:
    """
    Perform a rollout to fix typos in a review.
    """
    review, difficulty = prepare_review(step)
    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": (
                    "You are an excellent language model. Correct grammar and spelling mistakes in the review. "
                    "Return the corrected review only without any extra commentary."
                ),
            }
        ],
        metadata={
            "difficulty": difficulty,
            "step": step,
        },
        reward=0,
    )

    trajectory.messages_and_choices.append(
        {"role": "user", "content": review}
    )

    # Simulation of the correction (this should be replaced by actual model inference logic)
    corrected_review = review.replace("are", "is").replace("prodct", "product")

    # Add fake review correction attempt
    trajectory.messages_and_choices.append({"role": "assistant", "content": corrected_review})

    # Validate and update reward
    if "product" in corrected_review:
        trajectory.reward = 1
    else:
        trajectory.metrics["invalid_solution"] = 1.0
        trajectory.reward = -1

    return trajectory

