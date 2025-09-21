"""Rollout logic for recipe scaling ART environment."""
import json
import random
from typing import Any

import art
import weave
from openai import AsyncOpenAI

from env import RANDOM_SEED, get_recipe, scale_recipe


@weave.op
@art.retry(exceptions=(json.JSONDecodeError,))  # retry on parse errors
async def rollout(model: art.Model, step: int, config: dict[str, Any]) -> art.Trajectory:
    """Run one rollout: ask model to scale a recipe and compute reward."""
    # Seed randomness for reproducibility
    random.seed(RANDOM_SEED + step)

    # Extract scenario parameters
    recipe_name = config.get("recipe_name")
    target_servings = config.get("target_servings")
    base = get_recipe(recipe_name)
    true_scaled = scale_recipe(base, target_servings)

    # Setup client
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
                    "You are a recipe scaling assistant. "
                    "Scale a given recipe and respond with JSON list of {name, quantity, unit}."
                ),
            }
        ],
        metadata={
            "recipe_name": recipe_name,
            "step": step,
        },
        reward=0.0,
    )

    # Create user prompt
    prompt = (
        f"Scale recipe '{recipe_name}' from {base['original_servings']} to "
        f"{target_servings} servings. Base recipe: {json.dumps(base['ingredients'])}."
    )
    trajectory.messages_and_choices.append({"role": "user", "content": prompt})

    # Get model response
    response = await client.chat.completions.create(
        model=model.name,
        messages=trajectory.messages(),
        max_completion_tokens=config.get("max_completion_tokens", 256),
        temperature=config.get("temperature", 0.7),
        top_p=config.get("top_p", 0.9),
        stream=False,
    )
    choice = response.choices[0]
    content = choice.message.content
    trajectory.messages_and_choices.append(choice)

    # Validate JSON format
    try:
        scaled = json.loads(content)
    except json.JSONDecodeError:
        trajectory.metadata["invalid_format"] = 1
        trajectory.metrics["invalid_solution"] = 1.0
        trajectory.reward = -1.0
        return trajectory

    # Validate structure and compute errors
    true_map = {ing['name']: ing['quantity'] for ing in true_scaled}
    resp_map = {}
    for item in scaled:
        name = item.get('name')
        qty = item.get('quantity')
        resp_map[name] = qty

    missing = [n for n in true_map if n not in resp_map]
    extra = [n for n in resp_map if n not in true_map]
    missing_flag = 1.0 if missing else 0.0
    extra_flag = 1.0 if extra else 0.0

    # Compute average relative error
    errors = []
    for name, true_q in true_map.items():
        resp_q = resp_map.get(name)
        if resp_q is None:
            continue
        errors.append(abs(resp_q - true_q) / true_q)
    avg_error = sum(errors) / len(errors) if errors else 1.0

    # Assign reward
    if missing_flag or extra_flag:
        reward = -1.0
    elif avg_error == 0.0:
        reward = 1.0
    elif avg_error < 0.05:
        reward = 0.5
    else:
        reward = -1.0

    # Record metrics and metadata
    trajectory.metrics["avg_error"] = avg_error
    trajectory.metrics["missing_flag"] = missing_flag
    trajectory.metrics["extra_flag"] = extra_flag
    trajectory.reward = reward

    return trajectory
