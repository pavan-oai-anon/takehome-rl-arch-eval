"""
ART rollout logic for the arithmetic game 24.
"""
from typing import Any
import art
import weave
from env import generate_digits, validate_solution


@weave.op
@art.retry(exceptions=(ValueError,))
async def rollout(model: art.Model, step: int, config: dict[str, Any]) -> art.Trajectory:
    digits = generate_digits()
    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": (
                    "Your goal is to combine four random digits to form the number 24. "
                    "Use +, -, *, or /. Format the response as XML: <solution>(expression)</solution>."
                ),
            },
            {
                "role": "user",
                "content": f"Digits: {digits}"
            }
        ],
        metadata={
            "step": step,
            "invalid_solution": 0.0,
        },
        reward=0,
    )

    client = art.AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )

    chat_completion = await client.chat.completions.create(
        max_completion_tokens=config['max_completion_tokens'],
        messages=trajectory.messages(),
        model=config['model_name'],
        stream=False,
    )

    choice = chat_completion.choices[0]
    content = choice.message.content
    assert isinstance(content, str)
    trajectory.messages_and_choices.append(choice)

    try:
        solution = content.replace('<solution>', '').replace('</solution>', '').strip()
        if validate_solution(digits, solution):
            trajectory.reward = 1  # Assign appropriate reward here
        else:
            trajectory.metadata["invalid_solution"] = 1.0
            trajectory.reward = -1
    except ValueError:
        trajectory.reward = -1
        trajectory.metadata["invalid_solution"] = 1.0

    return trajectory

