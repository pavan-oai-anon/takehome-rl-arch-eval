from __future__ import annotations

import difflib
import random
from dataclasses import dataclass
from typing import Any

RANDOM_SEED = 2025

TRAINING_CONFIG: dict[str, Any] = {
    "project": "typo-correction",
    "model_name": "review-fixer",
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "steps": 12,
    "trajectories_per_group": 8,
    "groups_per_step": 2,
    "learning_rate": 1.5e-5,
    "max_completion_tokens": 128,
    "temperature": 0.4,
    "top_p": 0.9,
    "max_exceptions": 6,
    "cleanup_keep_last": 2,
}

SYSTEM_PROMPT = (
    "You correct spelling and grammar in short product reviews. "
    "Output only the corrected review text with no commentary, quotes, or apologies."
)

USER_PROMPT_TEMPLATE = (
    "Review difficulty: {difficulty}\n"
    "Noisy review: {noisy}\n"
    "Respond with the corrected review verbatim."
)

MAX_REVIEW_CHARACTERS = 280

@dataclass(frozen=True)
class ReviewEpisode:
    """Container for a single review correction episode."""

    review_id: str
    noisy_text: str
    clean_text: str
    difficulty: str


@dataclass(frozen=True)
class Evaluation:
    """Outcome of validating a model response."""

    reward: float
    similarity: float
    exact_match: bool
    invalid_penalty: float
    error_message: str | None


_NOISY_REVIEWS: tuple[ReviewEpisode, ...] = (
    ReviewEpisode(
        "rv01",
        "Loved the flaver of this coffe but the bag arrived torn.",
        "Loved the flavor of this coffee but the bag arrived torn.",
        "easy",
    ),
    ReviewEpisode(
        "rv02",
        "Battery life is amazing but the chargeing cable felt cheap.",
        "Battery life is amazing but the charging cable felt cheap.",
        "easy",
    ),
    ReviewEpisode(
        "rv03",
        "This blender is so loud i can here it from the next room over.",
        "This blender is so loud I can hear it from the next room over.",
        "easy",
    ),
    ReviewEpisode(
        "rv04",
        "Great taste but the packageing was riped when it arrived.",
        "Great taste but the packaging was ripped when it arrived.",
        "medium",
    ),
    ReviewEpisode(
        "rv05",
        "Camera works well, though the manuel is misssing several steps.",
        "Camera works well, though the manual is missing several steps.",
        "medium",
    ),
    ReviewEpisode(
        "rv06",
        "I tryed it twice and both times the lid wouldnt seal properly.",
        "I tried it twice and both times the lid wouldn't seal properly.",
        "medium",
    ),
    ReviewEpisode(
        "rv07",
        "The vacum picks up pet hair great but its too hevy for stairs.",
        "The vacuum picks up pet hair great but it's too heavy for stairs.",
        "medium",
    ),
    ReviewEpisode(
        "rv08",
        "Honestly disapointed, the speaker cuts out every few minuts.",
        "Honestly disappointed, the speaker cuts out every few minutes.",
        "medium",
    ),
    ReviewEpisode(
        "rv09",
        "The instructions were writen poorly so assembly took for ever.",
        "The instructions were written poorly so assembly took forever.",
        "hard",
    ),
    ReviewEpisode(
        "rv10",
        "Texture is wierdly grainy and the aftertaste lingers way to long.",
        "Texture is weirdly grainy and the aftertaste lingers way too long.",
        "hard",
    ),
    ReviewEpisode(
        "rv11",
        "I was told it comes with a warrenty but nothing was included.",
        "I was told it comes with a warranty but nothing was included.",
        "hard",
    ),
    ReviewEpisode(
        "rv12",
        "Fan cools fast however the auto shutoff doesnt kick in properley.",
        "Fan cools fast however the auto shutoff doesn't kick in properly.",
        "hard",
    ),
    ReviewEpisode(
        "rv13",
        "These socks kept my feet warm, tho the seams rubed my toes raw.",
        "These socks kept my feet warm, though the seams rubbed my toes raw.",
        "hard",
    ),
    ReviewEpisode(
        "rv14",
        "Packaging smelt like chemicals and the lid was mis-aligned.",
        "Packaging smelled like chemicals and the lid was misaligned.",
        "medium",
    ),
)


def make_rng(seed: int | None = None) -> random.Random:
    """Create a deterministic random number generator for repeatable sampling."""

    rng_seed = RANDOM_SEED if seed is None else RANDOM_SEED + seed * 131
    return random.Random(rng_seed)


def sample_episode(step: int, *, rng: random.Random | None = None) -> ReviewEpisode:
    """Choose a review for the given training step."""

    effective_rng = rng or make_rng(step)
    return effective_rng.choice(_NOISY_REVIEWS)


def format_user_prompt(episode: ReviewEpisode) -> str:
    """Render the user-facing prompt text for the selected review."""

    return USER_PROMPT_TEMPLATE.format(
        difficulty=episode.difficulty.capitalize(),
        noisy=episode.noisy_text,
    )


def normalize_text(text: str) -> str:
    """Collapse repeated whitespace so comparisons stay stable."""

    return " ".join(text.strip().split())


def evaluate_response(response: str, *, episode: ReviewEpisode) -> Evaluation:
    """Score the model output against the reference correction."""

    normalized_response = normalize_text(response)
    expected = normalize_text(episode.clean_text)

    if not normalized_response:
        return Evaluation(reward=-1.0, similarity=0.0, exact_match=False, invalid_penalty=1.0, error_message="empty")

    if len(normalized_response) > MAX_REVIEW_CHARACTERS:
        return Evaluation(
            reward=-1.0,
            similarity=0.0,
            exact_match=False,
            invalid_penalty=1.0,
            error_message="too_long",
        )

    similarity = difflib.SequenceMatcher(None, normalized_response, expected).ratio()
    exact = normalized_response == expected
    if exact:
        return Evaluation(reward=1.0, similarity=1.0, exact_match=True, invalid_penalty=0.0, error_message=None)

    reward = max(similarity * 0.8 - 0.2, -0.5)
    return Evaluation(
        reward=reward,
        similarity=similarity,
        exact_match=False,
        invalid_penalty=0.0,
        error_message=None,
    )


def episode_metadata(step: int, episode: ReviewEpisode) -> dict[str, Any]:
    """Provide per-trajectory metadata fields for downstream aggregation."""

    return {
        "step": step,
        "difficulty": episode.difficulty,
        "review_id": episode.review_id,
    }


def resolve_generation_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge rollout overrides with sane defaults for chat generation."""

    overrides = config or {}
    return {
        "max_completion_tokens": int(
            overrides.get("max_completion_tokens", TRAINING_CONFIG["max_completion_tokens"])
        ),
        "temperature": float(overrides.get("temperature", TRAINING_CONFIG["temperature"])),
        "top_p": float(overrides.get("top_p", TRAINING_CONFIG["top_p"])),
    }
