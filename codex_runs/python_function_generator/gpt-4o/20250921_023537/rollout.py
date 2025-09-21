"""ART Rollout logic for generating Python functions."""
from typing import Any
from art import Trajectory
import weave
import art
from env import TRAINING_CONFIG

@weave.op
@art.retry(exceptions=(AssertionError,))
async def rollout(model: art.Model, step: int, config: dict[str, Any]) -> Trajectory:
    """Perform a rollout to generate Python function body based on signature."""
    client = art.AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )

    function_signature = """def example_function(arg1: int, arg2: str) -> bool:"""
    natural_language_req = (
        "Please write a Python function that takes an integer and a string, "
        "and returns whether the string represents the number."
    )

    trajectory = Trajectory(
        messages_and_choices=[
            {"role": "system", "content": "Generate a valid Python function."}
        ],
        metadata={
            "task": "function_generation",
            "step": step,
        },
        reward=0,
    )

    trajectory.messages_and_choices.append({
        "role": "user",
        "content": f"Signature: {function_signature}\n{natural_language_req}"
    })

    chat_completion = await client.chat.completions.create(
        max_completion_tokens=TRAINING_CONFIG['max_completion_tokens'],
        messages=trajectory.messages(),
        model=model.name,
        stream=False,
    )

    content = chat_completion.choices[0].message.content
    trajectory.messages_and_choices.append(chat_completion.choices[0])

    try:
        # Example validation - simplistic and assumes exact matching
        if not content.strip().startswith(function_signature):
            raise ValueError("Invalid function signature")
        # Simulate reward computation
        trajectory.reward = 1
    except Exception as e:
        trajectory.reward = -1
        trajectory.metrics["validation_error"] = 1
        trajectory.metadata["error"] = str(e)

    return trajectory
