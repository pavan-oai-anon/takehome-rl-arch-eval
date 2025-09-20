"""ART rollout for JSON extraction task.

Assumes LocalBackend; GPU memory tuning handled externally."""
from __future__ import annotations

import json
import random
from typing import Any, Dict

import art
import weave
import requests
from openai import AsyncOpenAI

from env import RANDOM_SEED, sample_example

@weave.op
@art.retry(exceptions=(requests.ReadTimeout,))
async def rollout(model: art.Model, step: int, config: Dict[str, Any]) -> art.Trajectory:
    """Generate a trajectory extracting JSON from unstructured text."""
    # Deterministic sampling per step
    random.seed(RANDOM_SEED + step)
    example = sample_example()
    # Initialize client for LocalBackend via inference API
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )
    # Build trajectory
    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": (
                    "You are an agent extracting structured JSON. "
                    "Given text, output valid JSON with keys "
                    "`name` (string), `age` (integer), `location` (string)."
                ),
            }
        ],
        metadata={
            "example_id": example["metadata"]["id"],
            "step": step,
        },
        reward=0.0,
    )
    trajectory.messages_and_choices.append({"role": "user", "content": example["text"]})

    # Request model completion
    completion = await client.chat.completions.create(
        model=model.name,
        messages=trajectory.messages(),
        max_completion_tokens=config["max_completion_tokens"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        stream=False,
    )
    choice = completion.choices[0]
    content = choice.message.content
    trajectory.messages_and_choices.append(choice)

    # Validate JSON output
    try:
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("Output is not a JSON object")
    except (json.JSONDecodeError, ValueError):
        trajectory.metadata["validation_error"] = 1.0
        trajectory.metrics["invalid_solution"] = 1.0
        trajectory.reward = 0.0
        return trajectory

    # Schema validation
    expected: Dict[str, type] = {"name": str, "age": int, "location": str}
    correct = 0
    for key, typ in expected.items():
        if key in data and isinstance(data[key], typ):
            correct += 1
    # Record validation errors
    trajectory.metadata["validation_error"] = 1.0 if correct < len(expected) else 0.0

    # Metrics and reward (fraction of correct fields)
    trajectory.metrics["correct_fields"] = float(correct)
    trajectory.reward = correct / len(expected)
    return trajectory
