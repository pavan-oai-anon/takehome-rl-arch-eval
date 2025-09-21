"""Environment definitions for typo correction tasks."""

import random

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Training configuration for ART
TRAINING_CONFIG = {
    "project": "typo-correction",
    "model_name": "typo-corrector",
    "base_model": "gpt-3.5-turbo",
    "steps": 10,
    "trajectories_per_group": 16,
    "groups_per_step": 1,
    "learning_rate": 2e-5,
    "max_completion_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_exceptions": 4,
    "cleanup_keep_last": 2,
}

# Sample noisy reviews with ground-truth corrections
SAMPLE_REVIEWS = [
    {"id": "1", "noisy": "I absolutly love this!",       "clean": "I absolutely love this!",       "difficulty": 0},
    {"id": "2", "noisy": "The quallity is terrible",    "clean": "The quality is terrible",      "difficulty": 0},
    {"id": "3", "noisy": "Works well, no comlaints",   "clean": "Works well, no complaints",    "difficulty": 0},
    {"id": "4", "noisy": "Definately worth the price",  "clean": "Definitely worth the price",   "difficulty": 0},
    {"id": "5", "noisy": "I recieved my order late",    "clean": "I received my order late",      "difficulty": 0},
    {"id": "6", "noisy": "Battery lastes long time",   "clean": "Battery lasts long time",      "difficulty": 1},
    {"id": "7", "noisy": "Not as descripted",           "clean": "Not as described",            "difficulty": 1},
    {"id": "8", "noisy": "Fife stars, would buy again", "clean": "Five stars, would buy again",   "difficulty": 1},
    {"id": "9", "noisy": "Great custumer service",      "clean": "Great customer service",       "difficulty": 1},
    {"id": "10","noisy": "Packaging was brokwn",        "clean": "Packaging was broken",         "difficulty": 1},
]

def get_random_sample() -> dict:
    """Return a random review sample."""
    return random.choice(SAMPLE_REVIEWS)
