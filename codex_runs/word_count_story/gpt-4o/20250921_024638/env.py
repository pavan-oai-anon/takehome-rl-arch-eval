"""Environment configuration for ART reinforcement learning task."""
from typing import Dict

RANDOM_SEED = 42

TRAINING_CONFIG: Dict[str, any] = {
    "project": "short-story",
    "model_name": "story-gen-v1",
    "base_model": "openai/gpt-3.5-turbo",
    "steps": 100,
    "trajectories_per_group": 10,
    "groups_per_step": 2,
    "learning_rate": 5e-5,
    "max_completion_tokens": 50,  # Example value, adjust as needed
    "temperature": 0.7,
    "top_p": 0.9,
    "max_exceptions": 3,
    "cleanup_keep_last": 1,
}

PROMPTS = [
    {"theme": "adventure", "word_count": 10},
    {"theme": "mystery", "word_count": 15},
    # Add more prompts as needed
]

def get_prompt(index: int) -> Dict[str, int | str]:
    """Return a prompt description and required word count."""
    if 0 <= index < len(PROMPTS):
        return PROMPTS[index]
    raise IndexError("Prompt index out of range.")

import random
random.seed(RANDOM_SEED)
