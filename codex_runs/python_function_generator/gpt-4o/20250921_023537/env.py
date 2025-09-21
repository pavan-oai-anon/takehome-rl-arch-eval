"""Environment helpers and configuration for ART reinforcement learning task. """
import random

RANDOM_SEED = 42  # Seed for reproducibility

TRAINING_CONFIG = {
    "project": "python-function-generator",
    "model_name": "function-gen-model",
    "base_model": "Qwen/Qwen2.5-1.5B",
    "steps": 20,
    "trajectories_per_group": 10,
    "groups_per_step": 1,
    "learning_rate": 1e-4,
    "max_completion_tokens": 150,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_exceptions": 5,
    "cleanup_keep_last": 3,
}

def set_random_seed() -> None:
    """Sets the random seed for reproducibility."""
    random.seed(RANDOM_SEED)
