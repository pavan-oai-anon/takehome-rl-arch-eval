"""ART environment helpers for exact word-count micro-stories.

This module defines tunable hyperparameters, a small reference corpus of
theme/target/solution triples, and utilities for prompt construction and
validation used by the rollout.

Dependencies (install with uv):
    uv pip install openpipe-art weave openai requests

Notes:
- Assumes LocalBackend for inference/training in the host project.
- Keep metadata values as scalars only; lists/dicts belong in code, not metadata.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

# -------------------------
# Tunable environment knobs
# -------------------------
RANDOM_SEED: int = 7

# Default training config consumed by the host training script.
# Keep values modest to work on a single GPU or CPU. Learning-rate and
# memory-tuning mirrors spirit of the 2048 example (commentary only).
TRAINING_CONFIG: dict[str, Any] = {
    "project": "wc-micro-stories",
    "model_name": "agent-wc-001",
    "base_model": "Qwen/Qwen2.5-1.5B",  # small, CPU-friendly baseline
    "steps": 20,
    "trajectories_per_group": 16,
    "groups_per_step": 1,
    "learning_rate": 1e-5,
    "max_completion_tokens": 96,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_exceptions": 16,
    # Cleanup policy copied from the 2048 example behavior to save disk.
    "cleanup_keep_last": 1,
}


# -------------------------
# Reference tasks (10–20)
# -------------------------
@dataclass(frozen=True)
class ReferenceTask:
    """A seed example used for few-shot prompting and sampling tasks."""

    id: int
    theme: str
    target_words: int
    solution: str


_REFERENCE_TASKS: list[ReferenceTask] = [
    ReferenceTask(
        1,
        "haunted lighthouse",
        10,
        "Waves whispered; the lighthouse lamp blinked, guiding lost ghosts home.",
    ),
    ReferenceTask(
        2,
        "time-travel picnic",
        12,
        "We packed sandwiches, then returned yesterday to eat them still warm together.",
    ),
    ReferenceTask(
        3,
        "cyberpunk heist",
        15,
        "Neon rain hissed while our code cracked vaults; we stole names, not credits and freedom.",
    ),
    ReferenceTask(4, "desert oasis", 10, "Mirage shimmered, but the water remembered our thirsty names today."),
    ReferenceTask(
        5,
        "post-apocalyptic garden",
        20,
        "Among crumbling malls, seedlings conquered checkout lanes; we traded sunlight, compost, and stories instead of prices or fear and silence.",
    ),
    ReferenceTask(
        6,
        "lost space station",
        12,
        "Alarms slept; windows bloomed constellations; we remembered Earth's gravity by hugging tight.",
    ),
    ReferenceTask(
        7,
        "underwater library",
        15,
        "Shelves swayed like kelp; books breathed bubbles; stories surfaced whenever curious fins brushed spines gently.",
    ),
    ReferenceTask(8, "robot learns cooking", 10, "It measured love wrong, yet dinner tasted right to everyone."),
    ReferenceTask(
        9,
        "mountain village festival",
        12,
        "Lanterns climbed night slopes; drums echoed; elders danced lighter than ash tonight.",
    ),
    ReferenceTask(
        10,
        "whale astronomy",
        15,
        "Under radiant ice, whales mapped constellations by song; we learned directions listening backwards, patient together.",
    ),
    ReferenceTask(
        11,
        "enchanted train commute",
        12,
        "Every station changed seasons; my ticket punched snowflakes into pocket springtime daily.",
    ),
    ReferenceTask(
        12,
        "diplomacy with dragons",
        20,
        "Tea kettles calmed tempers; treaties inked in soot; we swapped hoard ledgers, promising interest payable in lullabies each winter solstice.",
    ),
    ReferenceTask(13, "haiku generator gone rogue", 10, "It counted carefully, then replaced syllables with honest confessions unexpectedly."),
    ReferenceTask(
        14,
        "friendly volcano",
        15,
        "It rumbled lullabies, knitting lava into warm roads; villagers baked bread on mornings smoky, cheerful.",
    ),
]


def reference_tasks() -> Sequence[ReferenceTask]:
    """Return the immutable list of seed tasks."""

    return tuple(_REFERENCE_TASKS)


def choose_task(step: int) -> ReferenceTask:
    """Deterministically choose a task for the given step.

    Uses a step-conditioned RNG so different steps explore different seeds
    while remaining reproducible across runs.
    """

    rnd = random.Random(RANDOM_SEED + step * 9973)
    return rnd.choice(_REFERENCE_TASKS)


# -------------------------
# Prompting and validation
# -------------------------
_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "by",
    "for",
    "with",
    "at",
    "into",
    "from",
    "as",
    "than",
    "then",
    "we",
    "our",
    "my",
    "your",
    "their",
    "it",
    "its",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "every",
    "each",
}


def build_system_prompt() -> str:
    """Concise system guidance for generating exact word-count micro-stories.

    Keeps the format expectations extremely clear to avoid formatting artifacts
    that would break word-count validation.
    """

    return (
        "You write tiny stories with an exact word count. "
        "Respond with plain text only: a single line, no numbering, no quotes, "
        "no code fences. Count words carefully before replying."
    )


def build_user_prompt(theme: str, target_words: int) -> str:
    """Construct the user instruction.

    Example: "Write a micro-story of exactly 12 words about: time-travel picnic."
    """

    return (
        f"Write a micro-story of exactly {target_words} words about: {theme}. "
        "Output plain text only."
    )


_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def word_count(text: str) -> int:
    """Count words by alphanumeric/apostrophe sequences, ignoring punctuation.

    Hyphenated tokens split into separate words ("time-travel" -> 2). Numbers and
    contractions count as words. Empty or whitespace-only text counts as 0.
    """

    if not text:
        return 0
    return len(_TOKEN_RE.findall(text))


def extract_keywords(theme: str, k: int = 4) -> list[str]:
    """Pick up to k salient keywords from the theme for coverage scoring.

    Naive approach: select non-stopword tokens by frequency/order.
    """

    tokens = [t.lower() for t in _TOKEN_RE.findall(theme)]
    keywords: list[str] = [t for t in tokens if t not in _STOPWORDS]
    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for w in keywords:
        if w not in seen:
            deduped.append(w)
            seen.add(w)
    return deduped[:k] if deduped else tokens[: min(k, len(tokens))]


def coverage_score(text: str, keywords: Iterable[str]) -> float:
    """Compute fraction of keywords present in text (0..1)."""

    if not keywords:
        return 0.0
    words = {w.lower() for w in _TOKEN_RE.findall(text)}
    keys = list(keywords)
    hits = sum(1 for k in keys if k in words)
    return hits / max(1, len(keys))


def validate_story(text: str, target_words: int) -> tuple[bool, str]:
    """Lightweight format validation.

    Returns (is_valid, error_message). Valid text is single-line plain text
    without code fences or JSON/XML markers.
    """

    if text is None:
        return False, "no_content"
    stripped = text.strip()
    if not stripped:
        return False, "empty"
    if "```" in stripped:
        return False, "code_fence"
    if stripped.startswith("{") or stripped.startswith("["):
        return False, "json_like"
    if "<" in stripped and ">" in stripped:
        return False, "xml_like"
    # Allow newlines but discourage them; single line is easier to count.
    if "\n" in stripped:
        return True, "newline"
    # Otherwise OK
    return True, "ok"


def few_shot_examples(rnd: random.Random, k: int = 2) -> list[tuple[str, str]]:
    """Return up to k (user, assistant) example pairs from the reference set."""

    examples = rnd.sample(_REFERENCE_TASKS, k=min(k, len(_REFERENCE_TASKS)))
    pairs: list[tuple[str, str]] = []
    for ex in examples:
        pairs.append((build_user_prompt(ex.theme, ex.target_words), ex.solution))
    return pairs


# End of env.py

