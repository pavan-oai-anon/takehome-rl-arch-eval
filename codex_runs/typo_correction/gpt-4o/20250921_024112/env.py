"""Environment setup for typo correction task using ART framework."""
import random

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Training configuration
TRAINING_CONFIG = {
    "project": "typo_correction",
    "model_name": "typo-fixer",
    "base_model": "base-model"
    "steps": 1000,
    "trajectories_per_group": 5,
    "groups_per_step": 3,
    "learning_rate": 1e-4,
    "max_completion_tokens": 50,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_exceptions": 3,
    "cleanup_keep_last": 1
}


def get_sample_reviews() -> list[str]:
    """Return a list of sample reviews with errors."""
    return [
        "The product are really good and cheap.",
        "This prodct definitely met all my expetations.",
        "Exellent customer service and very halpful.",
        "Will recommand to everyone I know.",
        "Decent quality but priccy for what you get."
    ]


def prepare_review(index: int) -> tuple[str, str]:
    """Return a review and its difficulty label."""
    reviews = get_sample_reviews()
    difficulty_levels = ["easy", "medium", "hard"]
    review = reviews[index % len(reviews)]
    difficulty = difficulty_levels[index % len(difficulty_levels)]
    return review, difficulty

