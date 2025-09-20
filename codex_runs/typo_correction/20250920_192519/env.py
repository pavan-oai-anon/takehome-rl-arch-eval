from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

RANDOM_SEED = 20240921
TRAINING_CONFIG = {
    "project": "typo-correction",
    "model_name": "typo-fixer",
    "base_model": "Qwen/Qwen2.5-3B",
    "steps": 12,
    "trajectories_per_group": 12,
    "groups_per_step": 1,
    "learning_rate": 5e-5,
    "max_completion_tokens": 96,
    "temperature": 0.3,
    "top_p": 0.9,
    "max_exceptions": 4,
    "cleanup_keep_last": 1,
}

SYSTEM_PROMPT = (
    "You correct typos in short product reviews. Respond with the corrected review "
    "only, without commentary or formatting cues."
)


@dataclass(frozen=True)
class ReviewEpisode:
    """Container for a single noisy review correction task."""

    episode_id: str
    noisy_review: str
    corrected_review: str
    difficulty: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of comparing a model response to the expected correction."""

    reward: float
    exact_match: bool
    invalid: bool
    error: str


_SAMPLE_REVIEWS: tuple[ReviewEpisode, ...] = (
    ReviewEpisode(
        episode_id="rev-001",
        noisy_review="This coffe maker brews relativly cold cups.",
        corrected_review="This coffee maker brews relatively cold cups.",
        difficulty="easy",
    ),
    ReviewEpisode(
        episode_id="rev-002",
        noisy_review="Battery life is amazzing though the screen scratches to easy.",
        corrected_review="Battery life is amazing though the screen scratches too easily.",
        difficulty="medium",
    ),
    ReviewEpisode(
        episode_id="rev-003",
        noisy_review="The blender came with a chiped blade and customer suport was slow.",
        corrected_review="The blender came with a chipped blade and customer support was slow.",
        difficulty="medium",
    ),
    ReviewEpisode(
        episode_id="rev-004",
        noisy_review="Great smell but the candle burns out in under a hour.",
        corrected_review="Great smell but the candle burns out in under an hour.",
        difficulty="easy",
    ),
    ReviewEpisode(
        episode_id="rev-005",
        noisy_review="Headphones sound grate but the left earcup squeeks when i move.",
        corrected_review="Headphones sound great but the left earcup squeaks when I move.",
        difficulty="medium",
    ),
    ReviewEpisode(
        episode_id="rev-006",
        noisy_review="I cant figure out the app, the instructions were poorly writen.",
        corrected_review="I can't figure out the app, the instructions were poorly written.",
        difficulty="hard",
    ),
    ReviewEpisode(
        episode_id="rev-007",
        noisy_review="Nice fabric yet the seams frayed affter two weeks of use.",
        corrected_review="Nice fabric yet the seams frayed after two weeks of use.",
        difficulty="medium",
    ),
    ReviewEpisode(
        episode_id="rev-008",
        noisy_review="The smart lock looses connection every few days and needs reset.",
        corrected_review="The smart lock loses connection every few days and needs a reset.",
        difficulty="hard",
    ),
    ReviewEpisode(
        episode_id="rev-009",
        noisy_review="Packiging was bent and the glass bottles arrived leeking.",
        corrected_review="Packaging was bent and the glass bottles arrived leaking.",
        difficulty="easy",
    ),
    ReviewEpisode(
        episode_id="rev-010",
        noisy_review="This toaster is fine but the crumb tray sticks everytime.",
        corrected_review="This toaster is fine but the crumb tray sticks every time.",
        difficulty="easy",
    ),
    ReviewEpisode(
        episode_id="rev-011",
        noisy_review="Camera picture quality is crisp though the menu's are confusing.",
        corrected_review="Camera picture quality is crisp though the menus are confusing.",
        difficulty="medium",
    ),
    ReviewEpisode(
        episode_id="rev-012",
        noisy_review="Keyboard feels solid, exept the spacebar rattles loudly.",
        corrected_review="Keyboard feels solid, except the spacebar rattles loudly.",
        difficulty="easy",
    ),
    ReviewEpisode(
        episode_id="rev-013",
        noisy_review="The humidifer makes a faint humm and leaves spots on the shelf.",
        corrected_review="The humidifier makes a faint hum and leaves spots on the shelf.",
        difficulty="medium",
    ),
    ReviewEpisode(
        episode_id="rev-014",
        noisy_review="I like the flavor but the teabags riped open in hot water.",
        corrected_review="I like the flavor but the teabags ripped open in hot water.",
        difficulty="easy",
    ),
    ReviewEpisode(
        episode_id="rev-015",
        noisy_review="Setup was confusing becuase the quick start guide skips steps.",
        corrected_review="Setup was confusing because the quick start guide skips steps.",
        difficulty="hard",
    ),
)


def get_rng(seed: int | None = None) -> random.Random:
    """Return a deterministic random generator keyed by the provided seed."""

    return random.Random(RANDOM_SEED if seed is None else seed)


def sample_episode(step: int, *, attempt: int = 0) -> ReviewEpisode:
    """Select a reproducible review episode for the given training step."""

    seed = RANDOM_SEED + (step * 9973) + (attempt * 7919)
    rng = get_rng(seed)
    return rng.choice(_SAMPLE_REVIEWS)


def render_user_prompt(episode: ReviewEpisode) -> str:
    """Create the user-facing prompt that describes the noisy review."""

    if episode.difficulty:
        return f"Difficulty: {episode.difficulty}\nNoisy review: {episode.noisy_review}"
    return f"Noisy review: {episode.noisy_review}"


def validate_response(response: str, episode: ReviewEpisode) -> ValidationResult:
    """Compare a model response with the canonical correction."""

    stripped = response.strip()
    if not stripped:
        return ValidationResult(
            reward=-1.0,
            exact_match=False,
            invalid=True,
            error="empty response",
        )

    if stripped != episode.corrected_review:
        error = "mismatch"
        if "\n" in stripped:
            error = "multiline mismatch"
        return ValidationResult(
            reward=-0.3,
            exact_match=False,
            invalid=True,
            error=error,
        )

    return ValidationResult(
        reward=1.0,
        exact_match=True,
        invalid=False,
        error="",
    )


def list_episode_ids() -> Iterable[str]:
    """Expose the known episode identifiers for quick inspection/testing."""

    return (episode.episode_id for episode in _SAMPLE_REVIEWS)
