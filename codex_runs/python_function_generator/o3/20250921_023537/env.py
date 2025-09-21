"""Environment helpers and shared constants for the ART function-generation task.

This module purposefully stays lightweight – only the absolute minimum that the
rollout implementation needs is exposed at import time.  Hyper-parameters and
task fixtures live at the top of the file so that tweaking them later is
straight-forward.
"""

from __future__ import annotations

import inspect
import random
import textwrap
from dataclasses import dataclass
from types import FunctionType
from typing import Any, Callable, List, Tuple

# ---------------------------------------------------------------------------
# Reproducibility & training hyper-parameters (easy to tweak)
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42

# NOTE: Training config is consumed by the generic training loop shipped with
# OpenPipe's ART framework – *not* by this task package directly.
# Keep scalar / JSON-serialisable values only.

TRAINING_CONFIG: dict[str, Any] = {
    "project": "function-gen",
    "model_name": "func-agent-001",
    "base_model": "Qwen/Qwen2.5-1.5B",
    "steps": 20,
    "trajectories_per_group": 16,
    "groups_per_step": 1,
    "learning_rate": 1e-5,
    "max_completion_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_exceptions": 16,
    "cleanup_keep_last": 1,
}


# ---------------------------------------------------------------------------
# Task dataset
# ---------------------------------------------------------------------------

# Each *episode* presents one `FunctionTask` to the agent.  We ship a small seed
# set so the reward function has deterministic ground truth to compare against.


@dataclass(frozen=True, slots=True)
class FunctionTask:
    """Container describing a single function-generation exercise."""

    name: str
    description: str  # Natural-language requirement shown to the agent.
    signature: str  # e.g. "def factorial(n: int) -> int:"
    # Callable returning a list of (args, expected_output) tuples used for
    # automatic evaluation.  Each `args` entry is itself a tuple of positional
    # arguments.
    tests: Callable[[], List[Tuple[Tuple[Any, ...], Any]]]
    # Reference implementation used for expected outputs and – if desired – as
    # a correctness oracle.
    golden_impl: FunctionType

    # helper ---------------------------------------------------------------
    def rendered_prompt(self) -> str:
        """Return the user prompt given to the agent."""

        return (
            f"Requirement: {self.description}\n"
            f"You must implement the following function signature.\n\n"
            f"{self.signature}\n"
            "Return *only* the complete function definition in Python, without "
            "additional comments or markdown fences. The body must adhere to "
            "PEP 8 and be self-contained.\n"
        )


def _factorial_impl(n: int) -> int:  # noqa: D401 – simple helpers
    """Return n! using an iterative approach."""

    acc = 1
    for i in range(2, n + 1):
        acc *= i
    return acc


def _is_prime_impl(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    k = 3
    while k * k <= n:
        if n % k == 0:
            return False
        k += 2
    return True


def _reverse_string_impl(s: str) -> str:
    return s[::-1]


def _fibonacci_impl(n: int) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


def _gcd_impl(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def _flatten_list_impl(nested: list[Any]) -> list[Any]:
    flat: list[Any] = []

    def _walk(seq: list[Any]) -> None:
        for item in seq:
            if isinstance(item, list):
                _walk(item)
            else:
                flat.append(item)

    _walk(nested)
    return flat


# Note: We purposely keep test cases tiny so that evaluation is lightning fast.


TASKS: list[FunctionTask] = [
    FunctionTask(
        name="factorial",
        description="Compute the factorial of a non-negative integer using iteration.",
        signature="def factorial(n: int) -> int:",
        tests=lambda: [((0,), 1), ((1,), 1), ((5,), 120)],
        golden_impl=_factorial_impl,
    ),
    FunctionTask(
        name="is_prime",
        description="Determine whether a given integer is a prime number.",
        signature="def is_prime(n: int) -> bool:",
        tests=lambda: [((2,), True), ((15,), False), ((17,), True)],
        golden_impl=_is_prime_impl,
    ),
    FunctionTask(
        name="reverse_string",
        description="Return the reversed version of the input string.",
        signature="def reverse_string(s: str) -> str:",
        tests=lambda: [(("hello",), "olleh"), (("",), "")],
        golden_impl=_reverse_string_impl,  # type: ignore[arg-type]
    ),
    FunctionTask(
        name="fibonacci",
        description="Return the nth Fibonacci number (1-indexed).",
        signature="def fibonacci(n: int) -> int:",
        tests=lambda: [((1,), 1), ((6,), 8)],
        golden_impl=_fibonacci_impl,
    ),
    FunctionTask(
        name="gcd",
        description="Compute the greatest common divisor of two integers.",
        signature="def gcd(a: int, b: int) -> int:",
        tests=lambda: [((54, 24), 6), ((0, 5), 5)],
        golden_impl=_gcd_impl,
    ),
    FunctionTask(
        name="flatten_list",
        description="Flatten a nested list into a single list of values (depth-first).",
        signature="def flatten_list(nested: list[Any]) -> list[Any]:",
        tests=lambda: [(([1, [2, 3]],), [1, 2, 3]), (([],), [])],
        golden_impl=_flatten_list_impl,
    ),
]


# ---------------------------------------------------------------------------
# Helper utilities used by rollout.py
# ---------------------------------------------------------------------------


def pick_task(step: int) -> FunctionTask:
    """Deterministically select a task based on the training step."""

    random.seed(RANDOM_SEED + step)
    return random.choice(TASKS)


def run_tests(func: FunctionType, task: FunctionTask) -> tuple[int, int]:
    """Run the agent-provided *func* against the golden test cases.

    Returns a tuple *(passed, total).* Any exception counts as a failure.
    """

    passed = 0
    test_cases = task.tests()

    for args, expected in test_cases:
        try:
            if not isinstance(args, tuple):  # For backwards compatibility.
                args = (args,)
            result = func(*args)
            if result == expected:
                passed += 1
        except Exception:  # pragma: no cover – defensive against bad outputs.
            continue
    return passed, len(test_cases)


def verify_signature(func: FunctionType, task: FunctionTask) -> bool:
    """Return *True* if *func* matches the requested signature."""

    # Cheap string-comparison of the call-signature to avoid heavy ast diffing.
    wanted = textwrap.dedent(task.signature).strip()
    actual = f"def {func.__name__}{inspect.signature(func)}:"
    return wanted == actual
