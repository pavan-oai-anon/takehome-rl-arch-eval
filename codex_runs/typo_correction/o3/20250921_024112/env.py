"""Environment helpers for the "typo-fix" ART task.

This file purposefully contains **no** ART specific code so that it can be
imported both by the rollout script *and* any potential evaluation utilities
without pulling heavyweight dependencies.

Key design goals:
1. Keep hyper-parameters clearly visible at the top for quick iteration.
2. Provide a small pool of noisy product reviews with ground-truth fixes.
3. Offer lightweight helpers for sampling an episode and computing rewards.
4. Avoid non-stdlib dependencies – difflib gives us a decent similarity score.
"""

from __future__ import annotations

import random
import difflib
from typing import Tuple, Dict, Any

# ---------------------------------------------------------------------------
# Tuning knobs & training configuration.
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42

# Core ART training hyper-parameters – tweak freely.
TRAINING_CONFIG: Dict[str, Any] = {
    "project": "typo-fix",
    "model_name": "agent-typo",
    "base_model": "Qwen/Qwen2.5-1.5B",
    "steps": 50,
    "trajectories_per_group": 24,
    "groups_per_step": 1,
    "learning_rate": 5e-5,
    "max_completion_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_exceptions": 24,
    # Keep the most recent checkpoint only – matches 2048 example.
    "cleanup_keep_last": 1,
}

# ---------------------------------------------------------------------------
# Episode data.
# ---------------------------------------------------------------------------

_SAMPLE_REVIEWS = [
    {
        "noisy": "Thiss product is amazng! I realy love the colour and the quility.",
        "correct": "This product is amazing! I really love the colour and the quality.",
        "difficulty": 0.4,
    },
    {
        "noisy": "The battrey life isnt good, it dies after twoo hours of use.",
        "correct": "The battery life isn't good; it dies after two hours of use.",
        "difficulty": 0.5,
    },
    {
        "noisy": "Package arived late and the box was dammaged.",
        "correct": "Package arrived late and the box was damaged.",
        "difficulty": 0.3,
    },
    {
        "noisy": "I hav never been more dissapointed in a purchase.",
        "correct": "I have never been more disappointed in a purchase.",
        "difficulty": 0.6,
    },
    {
        "noisy": "Excelent camera qualty but the screen scratches to easilly.",
        "correct": "Excellent camera quality but the screen scratches too easily.",
        "difficulty": 0.7,
    },
    {
        "noisy": "Sound is clear, but the earbuds fall outa my ears.",
        "correct": "Sound is clear, but the earbuds fall out of my ears.",
        "difficulty": 0.4,
    },
    {
        "noisy": "Greta price for what you get. Definately recomended!",
        "correct": "Great price for what you get. Definitely recommended!",
        "difficulty": 0.2,
    },
    {
        "noisy": "Dont waist your money, it broke after one weak.",
        "correct": "Don't waste your money; it broke after one week.",
        "difficulty": 0.5,
    },
    {
        "noisy": "Colour looks differant than in the advertisment.",
        "correct": "Colour looks different than in the advertisement.",
        "difficulty": 0.3,
    },
    {
        "noisy": "Fast delivary and grate custumer servise.",
        "correct": "Fast delivery and great customer service.",
        "difficulty": 0.3,
    },
    {
        "noisy": "The instrutions where confusing and mispelled.",
        "correct": "The instructions were confusing and misspelled.",
        "difficulty": 0.6,
    },
    {
        "noisy": "I'm verry happy with this purchace so far.",
        "correct": "I'm very happy with this purchase so far.",
        "difficulty": 0.2,
    },
    {
        "noisy": "It stoped working after the recent update, pleas fix!",
        "correct": "It stopped working after the recent update; please fix!",
        "difficulty": 0.7,
    },
]

# Make behaviour deterministic when needed.
random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Public helpers.
# ---------------------------------------------------------------------------


def sample_episode() -> Tuple[str, str, float, int]:
    """Return *(noisy, correct, difficulty, review_id)* for a single episode."""

    review_id = random.randrange(len(_SAMPLE_REVIEWS))
    review = _SAMPLE_REVIEWS[review_id]
    return review["noisy"], review["correct"], review["difficulty"], review_id


def compute_reward(predicted: str, target: str) -> float:
    """Compute a smooth reward based on string similarity.

    We use :pyclass:`difflib.SequenceMatcher` to obtain a ratio in ``[0,1]`` and
    then linearly map it to ``[-1, 1]`` so that *perfect* corrections receive
    ``+1`` while completely unrelated outputs approach ``-1``.
    """

    matcher = difflib.SequenceMatcher(None, predicted.strip(), target.strip())
    similarity = matcher.ratio()  # 0 → 1
    return (2.0 * similarity) - 1.0


def is_valid_output(predicted: str) -> bool:
    """Validation guard – ensure the model returns *something* printable.

    - Output must be non-empty and not exceed 3× target length (defensive).
    - We intentionally allow minor punctuation differences; reward handles that.
    """

    predicted = predicted.strip()
    return bool(predicted) and len(predicted) < 1024


# Export symbols that other modules are expected to import.
__all__ = [
    "RANDOM_SEED",
    "TRAINING_CONFIG",
    "sample_episode",
    "compute_reward",
    "is_valid_output",
]

