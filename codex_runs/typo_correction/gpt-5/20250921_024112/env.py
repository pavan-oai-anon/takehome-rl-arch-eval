"""Minimal ART environment for typo-correction on short product reviews.

This module exposes:
- RANDOM_SEED: int used for deterministic sampling
- TRAINING_CONFIG: dict with common training knobs
- Small dataset of noisy -> clean review pairs with difficulty labels
- Helper utilities for prompt building, validation, and reward shaping

Dependencies (install if needed):
- uv pip install openai weave requests

Notes:
- Designed for LocalBackend training/inference with small models.
- Keep hyperparameters compact and tweakable at the top.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
import re
from typing import Any, Iterable


# -----------------------------
# Tunable Environment Constants
# -----------------------------
RANDOM_SEED: int = 17

# Default training knobs consumed by the host training loop.
# These are conservative and memory-friendly; adjust per hardware.
TRAINING_CONFIG: dict[str, Any] = {
    "project": "typo-correction-reviews",
    "model_name": "typo-agent-001",
    "base_model": "Qwen/Qwen2.5-1.5B",  # local-friendly baseline
    "steps": 12,
    "trajectories_per_group": 12,
    "groups_per_step": 1,
    "learning_rate": 7e-6,
    # Inference sampling
    "max_completion_tokens": 96,
    "temperature": 0.2,
    "top_p": 0.95,
    # Infra/runtime
    "max_exceptions": 12,
    "cleanup_keep_last": 1,
}


# -----------------------------
# Dataset
# -----------------------------
@dataclass(frozen=True)
class ReviewExample:
    """One supervised pair for typo correction.

    Attributes:
        rid: short string id
        noisy: noisy input review presented to the agent
        clean: ground-truth corrected review used for reward
        difficulty: "easy" | "medium" | "hard"
    """

    rid: str
    noisy: str
    clean: str
    difficulty: str


DATASET: tuple[ReviewExample, ...] = (
    ReviewExample(
        "r01",
        "The prodcut was amazng, I realy liked the qualty.",
        "The product was amazing, I really liked the quality.",
        "medium",
    ),
    ReviewExample(
        "r02",
        "Arrived late and pakage was torn. Not hapy.",
        "Arrived late and package was torn. Not happy.",
        "easy",
    ),
    ReviewExample(
        "r03",
        "Battery life is suprisingly good; lasts all day long.",
        "Battery life is surprisingly good; lasts all day long.",
        "easy",
    ),
    ReviewExample(
        "r04",
        "This is okay-ish, but dose not match the discription.",
        "This is okay-ish, but does not match the description.",
        "medium",
    ),
    ReviewExample(
        "r05",
        "Great value for the money, would buy again.",
        "Great value for the money. Would buy again.",
        "hard",
    ),
    ReviewExample(
        "r06",
        "The color is off; it look different than the photos.",
        "The color is off; it looks different than the photos.",
        "easy",
    ),
    ReviewExample(
        "r07",
        "Definately not worth it.",
        "Definitely not worth it.",
        "easy",
    ),
    ReviewExample(
        "r08",
        "Super fast delievery, well packed!",
        "Super fast delivery, well packed!",
        "easy",
    ),
    ReviewExample(
        "r09",
        "Ive used it for 2 weeks and its fine.",
        "I've used it for 2 weeks and it's fine.",
        "medium",
    ),
    ReviewExample(
        "r10",
        "Sound quailty is poor, lots of noize.",
        "Sound quality is poor, lots of noise.",
        "easy",
    ),
    ReviewExample(
        "r11",
        "Too expencive for what you get.",
        "Too expensive for what you get.",
        "easy",
    ),
    ReviewExample(
        "r12",
        "Brillant service and very frendly support team.",
        "Brilliant service and very friendly support team.",
        "easy",
    ),
    ReviewExample(
        "r13",
        "Size runs small, orderd a size up.",
        "Size runs small, ordered a size up.",
        "easy",
    ),
    ReviewExample(
        "r14",
        "Good build, but the instrucitons were confusing.",
        "Good build, but the instructions were confusing.",
        "easy",
    ),
    ReviewExample(
        "r15",
        "absolutly love it!",
        "Absolutely love it!",
        "easy",
    ),
    ReviewExample(
        "r16",
        "Came with scracthes on the screen",
        "Came with scratches on the screen.",
        "medium",
    ),
    ReviewExample(
        "r17",
        "Not as addvertised, missing accesories.",
        "Not as advertised, missing accessories.",
        "medium",
    ),
    ReviewExample(
        "r18",
        "The app keeps crashin on startup.",
        "The app keeps crashing on startup.",
        "easy",
    ),
    ReviewExample(
        "r19",
        "Worth the prise for the fetures.",
        "Worth the price for the features.",
        "easy",
    ),
    ReviewExample(
        "r20",
        "Packaging smelt wierd but product works fine.",
        "Packaging smelled weird but product works fine.",
        "medium",
    ),
)


# -----------------------------
# Selection & Prompt Helpers
# -----------------------------
def difficulty_code(label: str) -> int:
    """Map difficulty labels to a stable integer code.

    Returns: 1 for easy, 2 for medium, 3 for hard, 0 unknown.
    """

    table = {"easy": 1, "medium": 2, "hard": 3}
    return table.get(label.lower(), 0)


def select_example(step: int) -> ReviewExample:
    """Deterministically pick one example for the given training step.

    Uses a seeded PRNG so that the same step yields the same episode.
    """

    rng = random.Random(RANDOM_SEED + int(step))
    return rng.choice(DATASET)


SYSTEM_PROMPT: str = (
    "You correct short product reviews. Fix spelling and grammar only. "
    "Return the corrected review verbatim as plain text. No quotes, no extra words, "
    "no explanations."
)


def user_prompt(ex: ReviewExample) -> str:
    """Compose a compact user prompt with an optional difficulty hint."""

    return (
        "Review (noisy):\n"
        f"{ex.noisy}\n"
        f"Difficulty: {ex.difficulty}\n"
        "Respond with only the corrected review text."
    )


# -----------------------------
# Validation & Reward Shaping
# -----------------------------
_SQUEEZE_WS_RE = re.compile(r"\s+")


def canonicalize(text: str) -> str:
    """Normalize whitespace and quotes for robust comparison.

    - Strip outer whitespace
    - Collapse internal runs of whitespace
    - Normalize straight quotes
    """

    t = text.strip()
    t = t.replace("\u2019", "'").replace("\u2018", "'")
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    t = _SQUEEZE_WS_RE.sub(" ", t)
    return t


def detect_format_violation(text: str) -> bool:
    """Detect common violations of the "plain text only" constraint.

    Flags if the output looks like a meta-comment or includes code/markup.
    """

    t = text.strip()
    if not t:
        return True
    lowered = t.lower()
    if (
        lowered.startswith("corrected:")
        or lowered.startswith("fix:")
        or lowered.startswith("answer:")
        or "```" in t
        or t.startswith("<") and t.endswith(">")
    ):
        return True
    # Require a single-line review; multi-line often indicates commentary.
    if "\n" in t:
        return True
    return False


def _levenshtein(a: Iterable[str], b: Iterable[str]) -> int:
    """Compute Levenshtein distance over an iterable of tokens.

    Runs in O(len(a)*len(b)) time; sufficient for short reviews.
    """

    a_list = list(a)
    b_list = list(b)
    if not a_list:
        return len(b_list)
    if not b_list:
        return len(a_list)
    m, n = len(a_list), len(b_list)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        ai = a_list[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == b_list[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,      # deletion
                cur[j - 1] + 1,   # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev = cur
    return prev[n]


def _norm_edit_distance(pred: str, gold: str) -> tuple[float, float]:
    """Normalized edit distance for characters and whitespace-separated words."""

    p_c = list(pred)
    g_c = list(gold)
    char_den = max(len(g_c), 1)
    char_dist = _levenshtein(p_c, g_c) / char_den

    p_w = pred.split()
    g_w = gold.split()
    word_den = max(len(g_w), 1)
    word_dist = _levenshtein(p_w, g_w) / word_den
    return char_dist, word_dist


def compute_metrics_and_reward(output: str, target: str) -> tuple[float, dict[str, float], bool, str]:
    """Score the model output with smooth rewards and useful metrics.

    Reward design:
    - Base on character and word accuracy: 0.7 * char_acc + 0.3 * word_acc
    - +0.2 bonus for exact canonical match
    - -0.3 penalty for format violations (commentary/markup/newlines)
    - Strong penalty for empty output

    Returns: (reward, metrics, format_violation, validation_error_message)
    """

    # Canonical comparisons smooth over spacing and curly quotes.
    out_c = canonicalize(output)
    tgt_c = canonicalize(target)

    format_violation = detect_format_violation(output)
    empty = 1.0 if not out_c else 0.0

    char_nd, word_nd = _norm_edit_distance(out_c, tgt_c)
    char_acc = 1.0 - char_nd
    word_acc = 1.0 - word_nd
    exact_match = 1.0 if out_c == tgt_c else 0.0

    reward = 0.7 * char_acc + 0.3 * word_acc
    if exact_match:
        reward += 0.2
    if format_violation:
        reward -= 0.3
    if empty:
        reward = -1.0

    # Keep reward bounded for stability.
    reward = max(-1.0, min(reward, 1.5))

    metrics = {
        "char_accuracy": float(char_acc),
        "word_accuracy": float(word_acc),
        "exact_match": float(exact_match),
        "norm_char_edit_distance": float(char_nd),
        "norm_word_edit_distance": float(word_nd),
        "invalid_solution": 1.0 if (format_violation or bool(empty)) else 0.0,
        "output_len": float(len(output.strip())),
        "target_len": float(len(target.strip())),
        "length_delta": float(len(output.strip()) - len(target.strip())),
    }

    err = ""
    if empty:
        err = "empty_output"
    elif format_violation:
        err = "format_violation"

    return reward, metrics, format_violation, err


__all__ = [
    "RANDOM_SEED",
    "TRAINING_CONFIG",
    "ReviewExample",
    "DATASET",
    "difficulty_code",
    "select_example",
    "SYSTEM_PROMPT",
    "user_prompt",
    "canonicalize",
    "compute_metrics_and_reward",
]

