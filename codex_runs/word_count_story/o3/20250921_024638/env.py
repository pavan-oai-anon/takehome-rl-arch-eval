"""Environment helpers for the *Exact Word-Count Story* ART task.

This module contains minimal utilities required by OpenPipe's ART framework
to train a language-model agent that writes short stories containing **exactly**
a requested number of words.  All hyper-parameters live at the top of the file
for easy experimentation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Tuple

# ---------------------------------------------------------------------------
# Global constants & training configuration
# ---------------------------------------------------------------------------

# NOTE: Keep a single, deterministic random seed for *all* stochastic helpers so
# that unit tests and reproducibility are trivial.
RANDOM_SEED: int = 2025


# Training-loop settings consumed by ART's generic `training.py` script.  These
# keys mirror those found in the 2048 example and are required by the prompt.
TRAINING_CONFIG: dict = {
    # wandb / weave project slug.
    "project": "story-word-count",
    # Name of the trainable model (arbitrary).
    "model_name": "story-agent",
    # The frozen base model that will receive the LoRA adapters.
    # (The host machine must already have this model or know how to fetch it.)
    "base_model": "Qwen/Qwen2.5-1.5B",
    # Total gradient-update steps.
    "steps": 50,
    # How many trajectory groups we gather per step.
    "groups_per_step": 1,
    # How many rollouts in each group.
    "trajectories_per_group": 16,
    # Adam learning-rate schedule.  Copied from 2048 baseline for memory safety.
    "learning_rate": 1e-5,
    # Generation parameters – we keep them conservative so the agent does not
    # abuse the token budget.
    "max_completion_tokens": 64,
    "temperature": 0.7,
    "top_p": 0.9,
    # Robustness & housekeeping.
    "max_exceptions": 16,
    "cleanup_keep_last": 3,
}

# ---------------------------------------------------------------------------
# Scenario generation helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StoryScenario:
    """A single training / evaluation scenario.

    Attributes
    ----------
    theme:  Short text describing what the story should be about.
    word_count:  Exact number of words the agent must output.
    reference:  Optional reference answer (not used for reward but handy for
                debugging / dataset inspection).
    """

    theme: str
    word_count: int
    reference: str | None = None


# A *tiny* seed dataset so the agent receives diverse prompts early on.  In
# practice you can extend / replace this with thousands of human-authored
# examples – ART will happily scale.

_SEED_SCENARIOS: List[StoryScenario] = [
    StoryScenario("a sunrise over the ocean", 10, "Waves whispered secrets as golden sunbeams painted the awakening sea."),
    StoryScenario("lost in a haunted library", 15, "Dusty tomes murmured forgotten curses while candles flickered, guiding Clara through endless aisles."),
    StoryScenario("cyberpunk street market", 12, "Neon drenched vendors bartered hacked dreams beneath buzzing hover-bikes and rain."),
    StoryScenario("dragon negotiates peace", 20, "Scales shimmering, the ancient dragon addressed trembling kings, proposing treaties forged in embered wisdom and mutual respect."),
    StoryScenario("interstellar coffee shop", 11, "Aliens traded star-maps for espresso, discussing supernovas over fragrant crema."),
    StoryScenario("time-looping birthday", 14, "Every candle relit itself, trapping Jonah in cheerful repetition until gratitude broke the cycle."),
    StoryScenario("robot learns to paint", 13, "Metal fingers hesitated, then blossomed colors, discovering humanity in gentle brushstrokes."),
    StoryScenario("hidden forest portal", 16, "Mossy archway shimmered; Eva stepped through, exchanging mundane worries for luminescent blossoms and curious sprites."),
    StoryScenario("deserted carnival at midnight", 18, "Rusty rides creaked haunting lullabies while moonlit tickets fluttered like ghosts around Liam's cautious footsteps."),
    StoryScenario("astronaut's lonely orbit", 17, "Earth rose blue and silent; Commander Reyes whispered stories to the twinkling void for company."),
]


def sample_scenario(step: int | None = None) -> StoryScenario:
    """Return a pseudo-random scenario.

    Parameters
    ----------
    step:  Current training step – incorporated into the RNG seed so that all
           workers sample identical scenarios for deterministic debugging.
    """

    rng = random.Random(RANDOM_SEED + (step or 0))
    return rng.choice(_SEED_SCENARIOS)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def count_words(text: str) -> int:
    """Return the number of whitespace-separated words in *text*.

    The tokenizer is deliberately naive – we treat any contiguous sequence of
    non-space characters as a word.  This matches how most evaluation scripts
    count words and is fast enough for feedback-loop latencies.
    """

    return len([token for token in text.strip().split() if token])


def word_count_reward(actual: int, target: int) -> float:
    """Dense reward shaping for the word-count task.

    The agent receives a reward \in [-1, 1].  Perfect matches yield 1.0.  The
    reward decays linearly with the *relative* error so the gradient remains
    informative even for large mistakes.
    """

    if target <= 0:
        return -1.0  # Defensive – should never happen.
    diff = abs(actual - target)
    return max(1.0 - diff / target, -1.0)


# Ensure deterministic behaviour across importers.
random.seed(RANDOM_SEED)

