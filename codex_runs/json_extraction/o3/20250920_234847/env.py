"""Lightweight helpers for the JSON-extraction ART task.

This module purposefully mirrors the structure of the 2048 example but
focuses on turning free-form text that embeds metadata into a strict JSON
payload with a known schema.  All constants that a practitioner may want to
adjust (hyper-parameters, schema, random seed, etc.) live at the top of the
file for quick iteration.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Tuple

# ---------------------------------------------------------------------------
# Public constants expected by the host project.
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 13

# ART training hyper-parameters.  The host `training.py` file will import this
# dict verbatim, so keep the keys stable and values simple (JSON-serialisable).
TRAINING_CONFIG: dict[str, Any] = {
    "project": "json-extraction",
    "model_name": "agent-json-001",
    "base_model": "Qwen/Qwen2.5-1.5B",
    # ---- RL specific ----
    "steps": 15,
    "trajectories_per_group": 16,
    "groups_per_step": 2,
    "learning_rate": 5e-6,
    # ---- inference parameters ----
    "max_completion_tokens": 256,
    "temperature": 0.2,
    "top_p": 0.9,
    # ---- misc ----
    "max_exceptions": 16,
    "cleanup_keep_last": False,
}

# ---------------------------------------------------------------------------
# Task definition.
# ---------------------------------------------------------------------------

# The extraction schema is intentionally tiny yet non-trivial so that reward
# shaping can grade partial success.  All values are strings except `amount`,
# which should parse to a float so that we can detect numeric formatting
# errors.
SCHEMA: dict[str, str] = {
    "name": "string",  # Person or company name.
    "date": "YYYY-MM-DD",  # ISO-8601 date.
    "amount": "float",  # Positive decimal number.
}


class Sample:
    """Container for a single training example."""

    __slots__ = ("id", "text", "ground_truth")

    def __init__(self, id_: str, text: str, ground_truth: Dict[str, Any]):
        self.id = id_
        self.text = text
        self.ground_truth = ground_truth


# A mini-corpus of examples – feel free to extend.
_SAMPLES: list[Sample] = [
    Sample(
        "s1",
        (
            "Invoice 2023-04-01\n"
            "Customer: Acme Corp.\n"
            "Total amount due: 1234.56 USD.\n"
            "Please pay by 2023-05-01."
        ),
        {"name": "Acme Corp.", "date": "2023-04-01", "amount": 1234.56},
    ),
    Sample(
        "s2",
        (
            "On 2022-12-24 John Doe donated $42 to our charity. "
            "Thank you, John!"
        ),
        {"name": "John Doe", "date": "2022-12-24", "amount": 42.0},
    ),
    Sample(
        "s3",
        (
            "Receipt #: 555\nDate: 2023-08-15\n"
            "Payee: Widgetry LLC\nAmount paid: EUR 99,95"
        ),
        {"name": "Widgetry LLC", "date": "2023-08-15", "amount": 99.95},
    ),
    Sample(
        "s4",
        (
            "Payment confirmation – Jane Smith paid GBP 12.30 on 2024-01-05."
        ),
        {"name": "Jane Smith", "date": "2024-01-05", "amount": 12.30},
    ),
    Sample(
        "s5",
        (
            "We received seventeen dollars (17.00 USD) from Foobar Inc on 2021-07-07."
        ),
        {"name": "Foobar Inc", "date": "2021-07-07", "amount": 17.0},
    ),
    Sample(
        "s6",
        (
            "Transfer record: 2020/02/29 – Buyer: Alice Wonderland – Sum: 2500"
        ),
        {"name": "Alice Wonderland", "date": "2020-02-29", "amount": 2500.0},
    ),
    Sample(
        "s7",
        (
            "Thank you for your purchase, Bob! We charged your card $3.5 on 2019-11-01."
        ),
        {"name": "Bob", "date": "2019-11-01", "amount": 3.5},
    ),
    Sample(
        "s8",
        "Mary Poppins donated on 2018-03-03. Amount: 7",
        {"name": "Mary Poppins", "date": "2018-03-03", "amount": 7.0},
    ),
    Sample(
        "s9",
        (
            "Sale completed 2017-09-09 for 49.99 USD paid by Stark Industries."
        ),
        {"name": "Stark Industries", "date": "2017-09-09", "amount": 49.99},
    ),
    Sample(
        "s10",
        (
            "Order total 0.99EUR — customer: TinyCorp (2024-04-01)."
        ),
        {"name": "TinyCorp", "date": "2024-04-01", "amount": 0.99},
    ),
]


def random_sample() -> Sample:
    """Return a deterministic random sample respecting the global seed."""

    random.seed(RANDOM_SEED)
    return random.choice(_SAMPLES)


def sample_for_step(step: int) -> Sample:
    """Deterministically pick a sample for the given training *step*."""

    index = (step + RANDOM_SEED) % len(_SAMPLES)
    return _SAMPLES[index]


# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------


def grade_extraction(predicted: Dict[str, Any], truth: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """Compute a dense reward based on field-level accuracy.

    Returns (reward, metrics) where metrics are flat scalars suitable for
    `trajectory.metrics`.  Reward ranges 0-1 for partial matches and 2 on
    perfect extraction to provide a stronger optimisation signal.
    """

    correct = 0
    total = len(SCHEMA)

    for field, _spec in SCHEMA.items():
        if field not in predicted:
            continue
        # normalise basic types
        if field == "amount":
            try:
                pred_val = float(predicted[field])
            except Exception:
                continue
            if abs(pred_val - float(truth[field])) < 1e-2:
                correct += 1
        else:
            if str(predicted[field]).strip() == str(truth[field]):
                correct += 1

    reward = correct / total  # 0-1 range
    if correct == total:
        reward = 2.0  # bonus for perfect extraction

    return reward, {
        "correct_fields": float(correct),
        "total_fields": float(total),
    }

