"""
Environment helpers for the typo correction task.
"""
from __future__ import annotations

import random
from typing import TypedDict, List

# Reproducible randomness seed
RANDOM_SEED: int = 42

# Training configuration for ART
TRAINING_CONFIG: dict[str, object] = {
    "project": "review-correction",
    "model_name": "review-corrector-001",
    "base_model": "gpt-3.5-turbo",
    "steps": 10,
    "trajectories_per_group": 16,
    "groups_per_step": 1,
    "learning_rate": 1e-5,
    "max_completion_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_exceptions": 4,
    "cleanup_keep_last": 2,
}


class Review(TypedDict):
    """Structure for a noisy product review sample."""
    id: str
    text: str
    difficulty: int

# Seeded list of sample reviews with common typos
SAMPLE_REVIEWS: List[Review] = [
    {"id": "1", "text": "I loove this product, it works perfctly.", "difficulty": 1},
    {"id": "2", "text": "Terrible custmer service, never buying again!", "difficulty": 2},
    {"id": "3", "text": "This item exceeded my expectaions, great value.", "difficulty": 1},
    {"id": "4", "text": "Fast delivery, but the packge was damaged.", "difficulty": 2},
    {"id": "5", "text": "Good qualit and easy to use.", "difficulty": 1},
    {"id": "6", "text": "Not worth the mony, very disappointing.", "difficulty": 2},
    {"id": "7", "text": "Excelent build, but descriptin was misleading.", "difficulty": 2},
    {"id": "8", "text": "Absolutely love it! Highly recomend.", "difficulty": 1},
    {"id": "9", "text": "Color was not as shown on the websitte.", "difficulty": 2},
    {"id": "10", "text": "Battery life is awsome, lasts all day.", "difficulty": 1},
    {"id": "11", "text": "The instrctions were confusng and unclear.", "difficulty": 2},
    {"id": "12", "text": "Value for money, but shipping took forevr.", "difficulty": 1},
]

def sample_review(step: int) -> Review:
    """
    Return a reproducible random review sample based on the training step.
    """
    rnd = random.Random(RANDOM_SEED + step)
    return rnd.choice(SAMPLE_REVIEWS)
