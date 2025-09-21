"""Environment helpers and configurations for recipe scaling RL task."""
import random

RANDOM_SEED = 42

# Configuration for training, easy to tweak hyperparameters
TRAINING_CONFIG = {
    "project": "recipe-scaling",
    "model_name": "recipe-scaler",
    "base_model": "Qwen/Qwen2.5-1.5B",
    "steps": 10,
    "trajectories_per_group": 5,
    "groups_per_step": 1,
    "learning_rate": 1e-4,
    "max_completion_tokens": 150,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_exceptions": 3,
    "cleanup_keep_last": 1,
}

def setup_recipes() -> list[dict]:
    """Set up base recipes to use in the reinforcement learning task."""
    return [
        {
            "name": "Pancakes",
            "ingredients": [
                {"name": "flour", "quantity": 200, "unit": "grams"},
                {"name": "milk", "quantity": 300, "unit": "ml"},
                {"name": "egg", "quantity": 2, "unit": "counts"},
            ],
            "original_servings": 4
        },
        # Add more recipes as needed
    ]

random.seed(RANDOM_SEED)
