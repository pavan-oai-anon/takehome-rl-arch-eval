"""
Rollout logic for malicious code classification task.
"""
from __future__ import annotations

import json
from typing import Any

import art
import weave
from env import sample_snippet
from openai import AsyncOpenAI

# Note: Using LocalBackend for inference; for GPU memory tuning, see 2048 example.

@weave.op
@art.retry(exceptions=(Exception,))
async def rollout(model: art.Model, step: int, config: dict[str, Any]) -> art.Trajectory:
    """
    Perform one classification episode for a code snippet.

    Args:
        model: ART model for inference.
        step: Current training step (used for sampling).
        config: Hyperparameters including max_completion_tokens, temperature, top_p.

    Returns:
        An art.Trajectory with messages, metadata, reward, and metrics.
    """
    # Sample a deterministic snippet based on step
    snippet = sample_snippet(step)

    # Initialize client for chat completions
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )

    # Initialize trajectory
    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": (
                    "You are a security classifier. Determine if the provided code "
                    "snippet is malicious. Respond with valid JSON: {\"is_malicious\": true/false, "
                    "\"explanation\": \"short explanation\"}."
                ),
            }
        ],
        metadata={
            "snippet_id": snippet["id"],
            "language": snippet["language"],
            "filename": snippet["filename"],
            "step": step,
        },
        reward=0.0,
    )

    # Add user message with snippet context
    user_content = (
        f"Language: {snippet['language']}\n"
        f"Filename: {snippet['filename']}\n"
        "Code:\n"
        f"{snippet['code']}"
    )
    trajectory.messages_and_choices.append({"role": "user", "content": user_content})

    # Generate classification from model
    chat = await client.chat.completions.create(
        model=model.name,
        messages=trajectory.messages(),
        max_completion_tokens=config.get("max_completion_tokens", 64),
        temperature=config.get("temperature", 0.7),
        top_p=config.get("top_p", 0.9),
        stream=False,
    )
    choice = chat.choices[0]
    content = choice.message.content
    trajectory.messages_and_choices.append(choice)

    # Validate JSON response
    invalid = 0.0
    try:
        result = json.loads(content)
        if not isinstance(result, dict):
            raise ValueError("Response not a JSON object")
        if "is_malicious" not in result or "explanation" not in result:
            raise ValueError("Missing required fields")
        if not isinstance(result["is_malicious"], bool):
            raise ValueError("`is_malicious` not boolean")
        if not isinstance(result["explanation"], str):
            raise ValueError("`explanation` not string")
    except Exception:
        invalid = 1.0

    # Assign metrics and reward
    if invalid:
        trajectory.reward = -1.0
        trajectory.metrics["invalid_solution"] = invalid
    else:
        predicted = result["is_malicious"]
        label = snippet["label"]
        correct = 1.0 if predicted == label else 0.0
        trajectory.reward = correct
        trajectory.metrics["correct"] = correct
        trajectory.metrics["invalid_solution"] = 0.0

    return trajectory
