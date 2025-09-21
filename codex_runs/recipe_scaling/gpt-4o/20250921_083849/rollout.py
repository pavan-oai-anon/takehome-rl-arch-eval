"""Rollout logic for recipe scaling task."""
from typing import Any
import json
from env import setup_recipes, TRAINING_CONFIG
import weave
import art

@weave.op
@art.retry(exceptions=(ValueError,))
async def rollout(model: art.Model, step: int, config: dict[str, Any]) -> art.Trajectory:
    recipes = setup_recipes()
    selected_recipe = recipes[0]  # Simple choice for demonstration
    requested_servings = 8  # Example requested servings
    
    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": (
                    "You are an expert chef. Scale the following recipe to "
                    f"{requested_servings} servings. Provide the recipe as a JSON list of ingredients."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(selected_recipe)
            }
        ],
        metadata={
            "recipe_id": selected_recipe["name"],
            "step": step,
        },
        reward=0,
    )

    # Simulate agent response (for illustration)
    agent_response = json.dumps([
        {"name": "flour", "quantity": 400, "unit": "grams"},
        {"name": "milk", "quantity": 600, "unit": "ml"},
        {"name": "egg", "quantity": 4, "unit": "counts"},
    ])

    try:
        response = json.loads(agent_response)
        trajectory.messages_and_choices.append({"role": "assistant", "content": agent_response})

        # Validate and calculate reward
        reward, errors = validate_scaled_recipe(selected_recipe, response, requested_servings)
        trajectory.reward = reward
        trajectory.metadata.update(errors)
    except json.JSONDecodeError:
        trajectory.reward = -1
        trajectory.metadata["invalid_solution"] = 1.0

    return trajectory

def validate_scaled_recipe(original: dict, scaled: list[dict], servings: int) -> tuple[float, dict]:
    """Validate the scaled recipe and return reward and errors."""
    errors = {}
    total_reward = 1.0
    orig_servings = original["original_servings"]

    ingredient_map = {ing["name"]: ing for ing in original["ingredients"]}
    scale_factor = servings / orig_servings

    for ing in scaled:
        name = ing["name"]
        if name not in ingredient_map:
            errors["missing_ingredient"] = 1.0
            total_reward -= 1
        else:
            expected_quantity = ingredient_map[name]["quantity"] * scale_factor
            actual_quantity = ing["quantity"]
            percentage_error = abs(expected_quantity - actual_quantity) / expected_quantity

            errors[f"{name}_percentage_error"] = percentage_error

            if percentage_error <= 0.05:
                total_reward += 0.5
            else:
                total_reward -= 1

    return total_reward, errors

