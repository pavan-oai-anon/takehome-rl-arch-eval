"""Environment helpers for the ART arithmetic 24 task."""
from __future__ import annotations

import ast
import random
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from typing import Sequence, Tuple

RANDOM_SEED: int = 1337
TARGET_VALUE: int = 24
NUM_DIGITS: int = 4
DIGIT_POOL: Tuple[int, ...] = tuple(range(1, 10))
EPSILON: float = 1e-6

PROJECT_NAME = "art-24-game"
MODEL_NAME = "solver-24"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TRAINING_STEPS = 40
TRAJECTORIES_PER_GROUP = 8
GROUPS_PER_STEP = 1
LEARNING_RATE = 5e-5
MAX_COMPLETION_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.95
MAX_EXCEPTIONS = TRAJECTORIES_PER_GROUP
CLEANUP_KEEP_LAST = 2

TRAINING_CONFIG: dict[str, object] = {
    "project": PROJECT_NAME,
    "model_name": MODEL_NAME,
    "base_model": BASE_MODEL,
    "steps": TRAINING_STEPS,
    "trajectories_per_group": TRAJECTORIES_PER_GROUP,
    "groups_per_step": GROUPS_PER_STEP,
    "learning_rate": LEARNING_RATE,
    "max_completion_tokens": MAX_COMPLETION_TOKENS,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "max_exceptions": MAX_EXCEPTIONS,
    "cleanup_keep_last": CLEANUP_KEEP_LAST,
}

random.seed(RANDOM_SEED)


class ExpressionValidationError(ValueError):
    """Raised when the agent submits an invalid expression."""


@dataclass(frozen=True)
class Puzzle:
    """Container for a single 24-game puzzle."""

    digits: Tuple[int, ...]
    identifier: str
    target: int = TARGET_VALUE


def create_puzzle() -> Puzzle:
    """Return a fresh puzzle sampled from the digit pool."""

    digits = tuple(random.choice(DIGIT_POOL) for _ in range(NUM_DIGITS))
    identifier = "".join(str(d) for d in digits) + f"-{random.randint(0, 9999):04d}"
    return Puzzle(digits=digits, identifier=identifier)


def render_user_prompt(puzzle: Puzzle) -> str:
    """Build the concise user prompt shown to the agent."""

    digits_text = " ".join(str(d) for d in puzzle.digits)
    return (
        "You are playing the arithmetic 24 game.\n"
        f"Numbers: {digits_text}\n"
        "Combine each number exactly once with +, -, *, or / to make 24.\n"
        "Respond with XML like <solution>(3 * (4 + 4))</solution>."
    )


def extract_expression_xml(payload: str) -> str:
    """Parse the assistant response and return the embedded expression."""

    try:
        root = ET.fromstring(payload.strip())
    except ET.ParseError as exc:  # pragma: no cover - defensive
        raise ValueError("Response was not valid XML") from exc

    if root.tag != "solution":
        raise ValueError("Root tag must be <solution>.")

    expression = (root.text or "").strip()
    if not expression:
        raise ValueError("<solution> must contain an expression.")

    return expression


def evaluate_expression(expression: str, digits: Sequence[int]) -> tuple[float, float, bool]:
    """Evaluate the expression and inspect digit usage."""

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:  # pragma: no cover - defensive
        raise ExpressionValidationError("Expression is not valid Python syntax") from exc

    used_literals: list[int] = []

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if abs(right) <= EPSILON:
                    raise ExpressionValidationError("Division by zero is not allowed")
                return left / right
            raise ExpressionValidationError("Only +, -, *, / operations are allowed")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = _eval(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool):
                raise ExpressionValidationError("Boolean values are not permitted")
            if not isinstance(value, (int, float)):
                raise ExpressionValidationError("Only numeric constants are allowed")
            if abs(value - round(value)) > EPSILON:
                raise ExpressionValidationError("Only integer digits are allowed")
            literal = int(round(value))
            used_literals.append(literal)
            return float(literal)
        raise ExpressionValidationError("Unsupported syntax in expression")

    result = _eval(tree)

    digits_counter = Counter(digits)
    used_counter = Counter(used_literals)
    for literal, count in used_counter.items():
        if digits_counter[literal] < count:
            raise ExpressionValidationError("Expression used unavailable digits")

    total_digits = len(digits)
    matched = sum(min(used_counter[d], digits_counter[d]) for d in digits_counter)
    coverage = matched / total_digits if total_digits else 0.0
    uses_all = all(used_counter.get(d, 0) == digits_counter[d] for d in digits_counter)
    return result, coverage, uses_all


def score_solution_xml(payload: str, puzzle: Puzzle) -> tuple[float, dict[str, float], str | None]:
    """Score an XML payload and return reward, metrics, and optional error text."""

    metrics: dict[str, float] = {
        "distance_to_target": float(TARGET_VALUE),
        "digit_coverage": 0.0,
        "uses_all_digits": 0.0,
        "exact_match": 0.0,
        "invalid_solution": 0.0,
    }

    try:
        expression = extract_expression_xml(payload)
    except ValueError as err:
        metrics["invalid_solution"] = 1.0
        return -1.5, metrics, str(err)

    try:
        value, coverage, uses_all = evaluate_expression(expression, puzzle.digits)
    except ExpressionValidationError as err:
        metrics["invalid_solution"] = 1.0
        return -1.5, metrics, str(err)

    distance = abs(value - puzzle.target)
    exact_match = distance <= EPSILON

    metrics.update(
        {
            "distance_to_target": float(distance),
            "digit_coverage": float(coverage),
            "uses_all_digits": 1.0 if uses_all else 0.0,
            "exact_match": 1.0 if exact_match else 0.0,
            "invalid_solution": 0.0,
        }
    )

    if exact_match and uses_all:
        reward = 2.0
    elif exact_match:
        reward = 1.0
    else:
        reward = max(-1.0, -distance / TARGET_VALUE)

    return reward, metrics, None


__all__ = [
    "RANDOM_SEED",
    "TRAINING_CONFIG",
    "Puzzle",
    "create_puzzle",
    "render_user_prompt",
    "score_solution_xml",
]
