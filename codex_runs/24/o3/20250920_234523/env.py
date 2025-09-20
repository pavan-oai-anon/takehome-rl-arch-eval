"""Environment helpers and shared constants for the *24 Game* ART task.

The classic game "24" presents four digits (1–9).  A player must combine the
digits **exactly once each** using the arithmetic operators +, -, *, / and any
parentheses so that the resulting value equals 24.

This module centralises logic that is useful for both rollout generation and
unit-style validation/reward shaping so that the same rules are applied during
training and (potentially) evaluation scripts.
"""

from __future__ import annotations

import ast
import math
import operator as _op
import random
import string
from typing import Callable, Sequence, Tuple

# ---------------------------------------------------------------------------
# Global configuration ----------------------------------------------------------------
# ---------------------------------------------------------------------------

# Reproducibility seed used wherever this module relies on the global RNG.
RANDOM_SEED: int = 42

# Training hyper-parameters consumed by the host ART trainer.  These values
# should be *small* so that examples run quickly on commodity hardware while
# still demonstrating learning behaviour.  Tweak as required.

TRAINING_CONFIG: dict[str, object] = {
    # bookkeeping
    "project": "24-game-multi-turn",
    "model_name": "agent-24",
    "base_model": "Qwen/Qwen2.5-1.5B",

    # RL schedule
    "steps": 8,
    "trajectories_per_group": 16,
    "groups_per_step": 1,
    "learning_rate": 1e-5,

    # inference generation
    "max_completion_tokens": 64,
    "temperature": 0.7,
    "top_p": 0.9,

    # misc runtime knobs
    "max_exceptions": 16,
    "cleanup_keep_last": 1,  # keep final checkpoint only (copy of 2048.py)
}


# ---------------------------------------------------------------------------
# Puzzle generation helpers ---------------------------------------------------
# ---------------------------------------------------------------------------

_DIGITS: tuple[int, ...] = tuple(range(1, 10))  # 1–9 inclusive


def generate_puzzle(rng: random.Random | None = None) -> Tuple[int, int, int, int]:
    """Return a fresh 4-tuple of digits for a new episode.

    Digits are sampled *with* replacement from 1-9 which is the common
    convention for the game.
    """

    rng = rng or random
    return tuple(rng.choice(_DIGITS) for _ in range(4))  # type: ignore[return-value]


def puzzle_to_string(puzzle: Sequence[int]) -> str:
    """Render puzzle digits as a compact user prompt payload."""

    return " ".join(map(str, puzzle))


# ---------------------------------------------------------------------------
# Solution validation ---------------------------------------------------------
# ---------------------------------------------------------------------------


class _SafeEvaluator(ast.NodeVisitor):
    """AST visitor that safely evaluates arithmetic expressions.

    Only the following node types are permitted:
    - `BinOp` with operators in {Add, Sub, Mult, Div}
    - `UnaryOp` with `UAdd`/`USub` (needed for regex-less negative literals)
    - `Constant` wrapping an `int`/`float`
    The visitor returns the numeric result *and* a tuple of literal numbers
    encountered so we can verify that the provided digits match the puzzle.
    """


    _BIN_OPS: dict[type[ast.AST], Callable[[float, float], float]] = {
        ast.Add: _op.add,
        ast.Sub: _op.sub,
        ast.Mult: _op.mul,
        ast.Div: _op.truediv,
    }

    def visit(self, node: ast.AST) -> tuple[float, tuple[int, ...]]:  # type: ignore[override]
        if isinstance(node, ast.Expression):
            return self.visit(node.body)

        if isinstance(node, ast.BinOp):
            if type(node.op) not in self._BIN_OPS:
                raise ValueError("unsupported operator")
            left_val, left_nums = self.visit(node.left)
            right_val, right_nums = self.visit(node.right)
            func = self._BIN_OPS[type(node.op)]
            try:
                value = func(left_val, right_val)
            except ZeroDivisionError as exc:  # pragma: no cover – defensive
                raise ValueError("division by zero") from exc
            return value, (*left_nums, *right_nums)

        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.UAdd, ast.USub)):
                raise ValueError("unsupported unary op")
            operand_val, operand_nums = self.visit(node.operand)
            return (+operand_val if isinstance(node.op, ast.UAdd) else -operand_val), operand_nums

        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value), (int(node.value),)

        raise ValueError("invalid expression component")


def evaluate_expression(expr: str) -> tuple[float, tuple[int, ...]]:
    """Safely evaluate *expr* and return (value, numbers_used)."""

    tree = ast.parse(expr, mode="eval")
    evaluator = _SafeEvaluator()
    return evaluator.visit(tree)


def validate_solution(expr: str, puzzle: Sequence[int]) -> tuple[bool, float, str | None]:
    """Validate *expr* against *puzzle*.

    Returns a triple *(is_valid, value, error_msg)* where *value* is 0.0 if the
    expression failed to evaluate.
    """

    try:
        value, numbers_used = evaluate_expression(expr)
    except (ValueError, SyntaxError) as exc:
        return False, 0.0, str(exc)

    # Check digits usage exactly once each.
    if sorted(numbers_used) != sorted(puzzle):
        return False, value, "digits used do not match puzzle"

    return True, value, None


# ---------------------------------------------------------------------------
# Reward shaping --------------------------------------------------------------
# ---------------------------------------------------------------------------


TARGET: int = 24


def reward_from_value(value: float, valid: bool) -> float:
    """Translate *value* into a reward signal.

    We want a smooth signal around the target so the policy receives gradient
    information even when not solving the puzzle exactly.  We therefore use an
    **exponential shaping**: `reward = exp(-|24 - value| / 24)` which is 1.0 for
    perfect answers and approaches 0 as the distance grows.  Invalid solutions
    receive -1 to strongly discourage rule-breaking.
    """

    if not valid:
        return -1.0

    distance = abs(TARGET - value)
    return math.exp(-distance / TARGET)


# ---------------------------------------------------------------------------
# Misc utilities --------------------------------------------------------------
# ---------------------------------------------------------------------------


def random_id(length: int = 6, *, rng: random.Random | None = None) -> str:
    """Return an alphanumeric identifier useful for grouping trajectories."""

    rng = rng or random
    alphabet = string.ascii_letters + string.digits
    return "".join(rng.choices(alphabet, k=length))


# Set module-level RNG seed once imported so that *all* helpers behave
# deterministically given the same process.
random.seed(RANDOM_SEED)

