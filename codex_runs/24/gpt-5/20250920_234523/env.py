"""Minimal environment helpers for the arithmetic game "24" using ART.

This module exposes simple utilities used by rollout.py and a tweakable
configuration block that mirrors the ergonomics of the 2048 example.

Notes:
- Assume LocalBackend for inference/training; model memory knobs (e.g.,
  `gpu_memory_utilization`) should be copied from the 2048 example if needed.
- The host project supplies the training loop and evaluation entry point.
"""
from __future__ import annotations

import ast
import random
import string
from dataclasses import dataclass
from typing import Iterable, Tuple


# ------------------------------
# Tunables and environment constants
# ------------------------------
RANDOM_SEED: int = 42
TARGET_VALUE: int = 24
DIGITS_PER_EPISODE: int = 4
MIN_DIGIT: int = 1
MAX_DIGIT: int = 9
EPSILON: float = 1e-6  # Numeric tolerance for equality checks
CLIPPED_ERROR: float = 24.0  # For smooth reward shaping
EXACT_SOLUTION_REWARD: float = 2.0  # Match the "win = 2" scale from 2048


# ------------------------------
# Training configuration consumed by the host trainer
# ------------------------------
TRAINING_CONFIG: dict = {
    # Project + model identity
    "project": "game-24",
    "model_name": "agent-24",
    "base_model": "Qwen/Qwen2.5-1.5B",  # Small, fast default
    # Trainer knobs
    "steps": 10,
    "trajectories_per_group": 18,
    "groups_per_step": 1,
    "learning_rate": 1e-5,
    # Inference knobs
    "max_completion_tokens": 96,
    "temperature": 0.7,
    "top_p": 0.9,
    # Fault tolerance + cleanup
    "max_exceptions": 18,
    "cleanup_keep_last": 1,
}


# ------------------------------
# Helpers
# ------------------------------


def generate_episode_digits() -> Tuple[int, int, int, int]:
    """Return 4 random digits in [MIN_DIGIT, MAX_DIGIT].

    Uses Python's global RNG. Callers may seed via RANDOM_SEED if desired.
    """

    return tuple(random.randint(MIN_DIGIT, MAX_DIGIT) for _ in range(DIGITS_PER_EPISODE))  # type: ignore[return-value]


def digits_to_string(digits: Iterable[int]) -> str:
    """Return a compact string representation suitable for metadata.

    The metadata must avoid lists/dicts for aggregation, so we serialize
    the digits as a comma-separated string.
    """

    return ",".join(str(d) for d in digits)


def new_episode_id(length: int = 6) -> str:
    """Generate a short alphanumeric ID for metadata."""

    alphabet = string.ascii_letters + string.digits
    return "".join(random.choices(alphabet, k=length))


@dataclass
class EvalResult:
    """Container for a safely evaluated arithmetic expression.

    Attributes
    -----------
    value: float | None
        Numeric value if evaluation succeeded; None otherwise.
    numbers: tuple[int, ...]
        All integer literals encountered in the expression (in-order).
    op_count: int
        Number of binary operations used in the expression.
    error: str | None
        Human-readable error string if evaluation failed.
    """

    value: float | None
    numbers: Tuple[int, ...]
    op_count: int
    error: str | None


def _multiset_equal(a: Iterable[int], b: Iterable[int]) -> bool:
    """Return True if two iterables contain the same elements with counts.

    Implemented without importing collections.Counter to keep the file lean.
    """

    counts: dict[int, int] = {}
    for x in a:
        counts[x] = counts.get(x, 0) + 1
    for x in b:
        if x not in counts:
            return False
        counts[x] -= 1
        if counts[x] == 0:
            del counts[x]
    return not counts


def numbers_match_episode(numbers: Iterable[int], episode_digits: Iterable[int]) -> bool:
    """Check that the integer literals exactly match the provided digits."""

    return _multiset_equal(numbers, episode_digits)


def safe_eval_arithmetic(expr: str) -> EvalResult:
    """Safely evaluate a restricted arithmetic expression.

    Permitted syntax: integers (0-9), binary operators +, -, *, /, and parentheses.
    Disallows unary operators, function calls, names, and other nodes.
    Returns a value and metadata about numbers/op count, or an error string.
    """

    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError as exc:  # pragma: no cover - defensive
        return EvalResult(value=None, numbers=(), op_count=0, error="syntax_error")

    numbers: list[int] = []

    def eval_node(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return eval_node(n.body)
        if isinstance(n, ast.BinOp):
            left = eval_node(n.left)
            right = eval_node(n.right)
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if isinstance(n.op, ast.Div):
                if abs(right) <= EPSILON:
                    raise ZeroDivisionError
                return left / right
            raise ValueError("unsupported_operator")
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            value = int(n.value)
            if not (0 <= value <= 9):  # restrict to single digits
                raise ValueError("non_digit_literal")
            numbers.append(value)
            return float(value)
        # Parentheses are represented structurally in the AST; no explicit node.
        raise ValueError("unsupported_syntax")

    try:
        value = eval_node(node)
        # op_count equals (#numbers - 1) for a valid binary tree over numbers
        op_count = max(0, len(numbers) - 1)
        return EvalResult(value=value, numbers=tuple(numbers), op_count=op_count, error=None)
    except ZeroDivisionError:
        return EvalResult(value=None, numbers=tuple(numbers), op_count=max(0, len(numbers) - 1), error="division_by_zero")
    except ValueError as exc:  # unsupported syntax, non-digit, etc.
        return EvalResult(value=None, numbers=tuple(numbers), op_count=max(0, len(numbers) - 1), error=str(exc))


def render_system_prompt() -> str:
    """Concise system instructions for the policy.

    Keep this short so the model has headroom for completions during training.
    """

    return (
        "You play the arithmetic game '24'. Combine the four given digits "
        "using +, -, *, / and parentheses to make exactly 24. Use each digit "
        "exactly once. Return only XML like <solution>(3 * (4 + 4))</solution>."
    )


def render_user_prompt(d1: int, d2: int, d3: int, d4: int) -> str:
    """User prompt describing the current episode's digits and format rules."""

    return (
        f"Digits: {d1}, {d2}, {d3}, {d4}. Rules: use each digit once, only + - * / "
        f"and parentheses. Output a single <solution>EXPR</solution> tag with no extra text."
    )


