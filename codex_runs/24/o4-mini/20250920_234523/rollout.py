"""Rollout logic for the '24 game' using ART."""

import random
from typing import Any

import art
import requests
import weave
from openai import AsyncOpenAI

from env import (
    RANDOM_SEED,
    TRAINING_CONFIG,
    generate_digits,
    parse_solution,
    evaluate_expression,
)

# Note: LocalBackend configuration and GPU memory tuning should be applied in setup_model.
@weave.op
@art.retry(exceptions=(requests.ReadTimeout,))
async def rollout(model: art.Model, step: int, config: dict[str, Any]) -> art.Trajectory:
    """
    Perform one rollout episode for the 24 game.
    Generates four digits, prompts the model for a solution, and computes reward.
    """
    # Seed per step for reproducibility
    random.seed(RANDOM_SEED + step)
    digits = generate_digits()

    # Initialize ART trajectory
    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": (
                    "You are an expert at the '24 game'. Given exactly four digits, "
                    "return an XML solution like <solution>(3 * (4 + 4))</solution> "
                    "using each digit exactly once and operators +, -, *, /."
                ),
            }
        ],
        metadata={
            "step": step,
            "digit1": digits[0],
            "digit2": digits[1],
            "digit3": digits[2],
            "digit4": digits[3],
        },
        reward=0.0,
    )

    # Add the digits prompt
    user_prompt = f"Digits: {digits[0]}, {digits[1]}, {digits[2]}, {digits[3]}"
    trajectory.messages_and_choices.append({"role": "user", "content": user_prompt})

    # Call the language model
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )
    completion = await client.chat.completions.create(
        max_completion_tokens=config["max_completion_tokens"],
        messages=trajectory.messages(),
        model=model.name,
        temperature=config.get("temperature"),
        top_p=config.get("top_p"),
        stream=False,
    )
    choice = completion.choices[0]
    trajectory.messages_and_choices.append(choice)

    # Process and validate the solution
    content = choice.message.content
    try:
        expr = parse_solution(content)
        result = evaluate_expression(expr)
        # Record the numeric result
        trajectory.metrics["result"] = result
        # Smooth reward: 1.0 if exact, else linearly scaled down
        diff = abs(result - 24)
        reward = 1.0 if diff == 0 else max(0.0, 1.0 - diff / 24)
        trajectory.reward = reward
    except ValueError:
        # Invalid solution: penalize and flag
        trajectory.metrics["invalid_solution"] = 1.0
        trajectory.reward = -1.0

    return trajectory
