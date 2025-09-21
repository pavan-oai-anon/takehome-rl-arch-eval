"""Environment helpers for recipe scaling task."""
import random

# Random seed for reproducibility
RANDOM_SEED: int = 42

# Base recipes definition: name -> original servings and ingredients
RECIPES = {
    "pancakes": {
        "original_servings": 4,
        "ingredients": [
            {"name": "flour", "quantity": 200, "unit": "g"},
            {"name": "milk", "quantity": 300, "unit": "ml"},
            {"name": "egg", "quantity": 2, "unit": "pcs"},
            {"name": "sugar", "quantity": 50, "unit": "g"},
            {"name": "baking_powder", "quantity": 5, "unit": "g"},
        ],
    },
    # Additional recipes can be added here
}

def get_recipe(name: str) -> dict:
    """Retrieve a base recipe by name."""
    return RECIPES[name]

def scale_recipe(recipe: dict, target_servings: int) -> list[dict]:
    """Scale ingredients proportionally to target servings."""
    factor = target_servings / recipe["original_servings"]
    scaled: list[dict] = []
    for ing in recipe["ingredients"]:
        scaled.append({
            "name": ing["name"],
            "quantity": ing["quantity"] * factor,
            "unit": ing["unit"],
        })
    return scaled

# Configuration for ART training
TRAINING_CONFIG: dict = {
    "project": "recipe-scaling",
    "model_name": "scaling-agent",
    "base_model": "gpt-3.5-turbo",
    "steps": 100,
    "trajectories_per_group": 8,
    "groups_per_step": 1,
    "learning_rate": 1e-5,
    "max_completion_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_exceptions": 4,
    "cleanup_keep_last": 2,
}
