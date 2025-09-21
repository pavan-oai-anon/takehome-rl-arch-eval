"""
ART rollout logic for code snippet classification task.
"""
from typing import Any
import json
import art
import weave
from env import get_snippet, TRAINING_CONFIG



async def rollout(model: art.Model, step: int, config: dict[str, Any]) -> art.Trajectory:
    client = art.AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )

    snippet = get_snippet(step % len(TRAINING_CONFIG))
    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": (
                    "You are an agent detecting malicious code."
                    " Provide a JSON response with 'is_malicious' (bool) and 'explanation' (str)."
                ),
            }
        ],
        metadata={
            "language": snippet["language"],
            "filename": snippet["filename"],
            "step": step,
        },
        reward=0,
    )

    trajectory.messages_and_choices.append(
        {"role": "user", "content": snippet["code"]}
    )

    chat_completion = await client.chat.completions.create(
        max_completion_tokens=128,
        messages=trajectory.messages(),
        model=model.name,
        stream=False,
    )

    choice = chat_completion.choices[0]
    content = choice.message.content
    trajectory.messages_and_choices.append(choice)

    try:
        response = json.loads(content)
        if (
            isinstance(response, dict)
            and "is_malicious" in response
            and "explanation" in response
            and isinstance(response["is_malicious"], bool)
            and isinstance(response["explanation"], str)
        ):
            correct_label = snippet["is_malicious"]
            trajectory.reward = 1.0 if response["is_malicious"] == correct_label else 0.0
            trajectory.metrics["invalid_solution"] = 0.0
        else:
            raise ValueError("Invalid JSON structure")
    except (ValueError, json.JSONDecodeError):
        trajectory.reward = -1.0
        trajectory.metrics["invalid_solution"] = 1.0

    return trajectory
