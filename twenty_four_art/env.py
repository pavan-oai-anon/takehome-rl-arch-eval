"""Environment helpers for the 24 arithmetic game."""
from __future__ import annotations

import ast
import math
import operator
import random
import string
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Iterable

# Digits range follows the classic 24 game rules (1-9 inclusive).
DIGIT_MIN = 1
DIGIT_MAX = 9
TARGET_VALUE = 24
RANDOM_SEED = 1234


@dataclass
class Game24:
    """State container representing one puzzle instance."""

    id: str
    digits: list[int]
    target: int = TARGET_VALUE

    def sorted_digits(self) -> list[int]:
        """Return the digits as a sorted list for multiset comparisons."""

        return sorted(self.digits)


def generate_game(*, rng: random.Random | None = None) -> Game24:
    """Create a fresh 24 game with four random digits.

    Parameters
    ----------
    rng:
        Optional random generator. A shared RNG makes unit testing reproducible.
    """

    generator = rng or random
    digits = [generator.randint(DIGIT_MIN, DIGIT_MAX) for _ in range(4)]
    puzzle_id = "".join(generator.choices(string.ascii_lowercase + string.digits, k=6))
    return Game24(id=puzzle_id, digits=digits)


def render_puzzle(game: Game24) -> str:
    """Pretty-print a puzzle prompt for the agent."""

    digits_str = " ".join(str(digit) for digit in game.digits)
    return (
        "Numbers: "
        f"{digits_str}\n"
        "Combine the numbers with +, -, *, / operations."
    )


_ALLOWED_BINOPS: dict[type[ast.operator], Callable[[Fraction, Fraction], Fraction]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_ALLOWED_UNARYOPS: dict[type[ast.unaryop], Callable[[Fraction], Fraction]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class InvalidExpressionError(ValueError):
    """Exception explaining why a candidate solution failed validation."""


def _validate_node(node: ast.AST) -> None:
    """Ensure the parsed AST only contains safe nodes."""

    if isinstance(node, ast.Expression):
        _validate_node(node.body)
        return

    if isinstance(node, ast.BinOp):
        if type(node.op) not in _ALLOWED_BINOPS:
            raise InvalidExpressionError("Only +, -, *, / operations are allowed")
        _validate_node(node.left)
        _validate_node(node.right)
        return

    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _ALLOWED_UNARYOPS:
            raise InvalidExpressionError("Only + and - unary operators are allowed")
        _validate_node(node.operand)
        return

    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise InvalidExpressionError("Only numeric constants are allowed")
        return

    raise InvalidExpressionError("Unsupported syntax in expression")


def _evaluate_node(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body)

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)
        if op_type is ast.Div and right == 0:
            raise InvalidExpressionError("Division by zero is not allowed")
        return _ALLOWED_BINOPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        return _ALLOWED_UNARYOPS[op_type](_evaluate_node(node.operand))

    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool):
            raise InvalidExpressionError("Boolean literals are not valid numbers")
        return Fraction(value)

    raise InvalidExpressionError("Unexpected AST node during evaluation")


def _extract_constants(node: ast.AST) -> list[int]:
    """Collect integer constants contained inside the expression."""

    constants: list[int] = []

    for child in ast.walk(node):
        if isinstance(child, ast.Constant):
            if isinstance(child.value, bool):
                raise InvalidExpressionError("Boolean literals are not valid numbers")
            if isinstance(child.value, (int, float)):
                as_int = int(child.value)
                if not math.isclose(child.value, as_int, abs_tol=1e-9):
                    raise InvalidExpressionError("Only integer constants may be used")
                if not (DIGIT_MIN <= as_int <= DIGIT_MAX):
                    raise InvalidExpressionError(
                        "Constants must match the provided single-digit numbers"
                    )
                constants.append(as_int)
    return constants


def validate_solution_expression(expression: str, digits: Iterable[int]) -> Fraction:
    """Validate that an expression solves the puzzle and return its value.

    Parameters
    ----------
    expression:
        The arithmetic expression extracted from the XML payload.
    digits:
        The digits supplied by the environment; each must be used exactly once.
    """

    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:  # pragma: no cover - defensive guard
        raise InvalidExpressionError("Malformed expression") from exc

    _validate_node(parsed)

    constants = _extract_constants(parsed)
    required = sorted(int(d) for d in digits)
    if sorted(constants) != required:
        raise InvalidExpressionError(
            "Each provided digit must appear exactly once in the expression"
        )

    value = _evaluate_node(parsed)
    return value


def score_expression(value: Fraction, target: int = TARGET_VALUE) -> dict[str, float]:
    """Return shaping metrics derived from the candidate expression value."""

    as_float = float(value)
    difference = abs(as_float - target)
    inverse_error = 1.0 / (1.0 + difference)
    normalized_reward = 2.0 if difference == 0 else max(0.0, 1.0 - (difference / target))
    return {
        "value": as_float,
        "difference": difference,
        "inverse_error": inverse_error,
        "normalized_reward": normalized_reward,
    }


def render_solution(value: Fraction) -> str:
    """Format a solved value for logging/debugging output."""

    return f"Value: {float(value):.4f}"
