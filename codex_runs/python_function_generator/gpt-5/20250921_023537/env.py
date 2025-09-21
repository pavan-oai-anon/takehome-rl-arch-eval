"""Environment helpers for ART Python-function generation task.

This module defines a small bank of programming exercises, utilities to
render prompts, and training hyperparameters expected by the host trainer.

Only scalars should be used in metadata by the consumer (rollout.py).

Install runtime deps, if needed, with uv:
  uv pip install openai weave art
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Callable, Iterable


# ------------------------
# Tweakable configuration
# ------------------------
RANDOM_SEED: int = 7

# Minimal training configuration consumed by the host project.
TRAINING_CONFIG: dict[str, Any] = {
    "project": "py-func-gen",
    "model_name": "agent-funcgen-001",
    "base_model": "Qwen/Qwen2.5-1.5B",
    "steps": 40,
    "trajectories_per_group": 18,
    "groups_per_step": 1,
    "learning_rate": 1e-5,
    "max_completion_tokens": 512,
    "temperature": 0.3,
    "top_p": 0.9,
    "max_exceptions": 18,
    # Keep a couple of recent checkpoints. LocalBackend is assumed.
    "cleanup_keep_last": 2,
}


@dataclass(frozen=True)
class Signature:
    """Function signature specification used in prompts and validation."""

    name: str
    args: tuple[str, ...]
    returns: str | None = None
    doc_hint: str = ""  # Phrase to include in docstring's first line.

    def header_line(self) -> str:
        """Render a canonical def line without trailing docstring/body."""

        args_str = ", ".join(self.args)
        if self.returns:
            return f"def {self.name}({args_str}) -> {self.returns}:"
        return f"def {self.name}({args_str}):"


@dataclass
class Problem:
    """A single exercise with golden implementation and test cases.

    Tests are specified as positional-argument tuples. Expected outputs are
    computed by the golden implementation during evaluation, avoiding duplication.
    """

    pid: int
    name: str
    description: str
    signature: Signature
    golden: Callable[..., Any]
    cases: tuple[tuple[Any, ...], ...]


def _cases(*rows: Iterable[Any]) -> tuple[tuple[Any, ...], ...]:
    """Compact helper to build a tuple of argument tuples."""

    return tuple(tuple(r) for r in rows)


# ------------------------
# Golden implementations
# ------------------------


def g_factorial(n: int) -> int:
    if n < 2:
        return 1
    out = 1
    for i in range(2, n + 1):
        out *= i
    return out


def g_gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def g_is_palindrome(s: str) -> bool:
    filtered = [c.lower() for c in s if c.isalnum()]
    return filtered == list(reversed(filtered))


def g_fibonacci(n: int) -> int:
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


def g_reverse_words(s: str) -> str:
    return " ".join(reversed([w for w in s.split() if w]))


def g_run_length_encode(s: str) -> str:
    if not s:
        return ""
    out: list[str] = []
    curr = s[0]
    count = 1
    for ch in s[1:]:
        if ch == curr:
            count += 1
        else:
            out.append(f"{curr}{count}")
            curr, count = ch, 1
    out.append(f"{curr}{count}")
    return "".join(out)


def g_unique_in_order(seq: str) -> str:
    out: list[str] = []
    prev: str | None = None
    for c in seq:
        if c != prev:
            out.append(c)
            prev = c
    return "".join(out)


def g_roman_to_int(s: str) -> int:
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    prev = 0
    for ch in reversed(s):
        v = vals[ch]
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    return total


def g_int_to_roman(n: int) -> str:
    pairs = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ]
    out: list[str] = []
    x = n
    for v, sym in pairs:
        while x >= v:
            out.append(sym)
            x -= v
    return "".join(out)


def g_two_sum_indices(nums: list[int], target: int) -> tuple[int, int]:
    seen: dict[int, int] = {}
    for i, v in enumerate(nums):
        need = target - v
        if need in seen:
            j = seen[need]
            return (j, i) if j < i else (i, j)
        seen[v] = i
    raise ValueError("no solution")


def g_mean(nums: list[float]) -> float:
    return 0.0 if not nums else sum(nums) / len(nums)


def g_flatten(nested: list[Any]) -> list[Any]:
    out: list[Any] = []
    stack: list[Any] = [nested]
    while stack:
        cur = stack.pop()
        if isinstance(cur, (list, tuple)):
            stack.extend(reversed(list(cur)))
        else:
            out.append(cur)
    return out


def g_levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost,    # substitution
            )
            prev = cur
    return dp[n]


def g_is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


# ------------------------
# Problem bank
# ------------------------


def build_problem_bank() -> tuple[Problem, ...]:
    """Return a stable tuple of problems across diverse skills."""

    problems: list[Problem] = []
    add = problems.append

    add(
        Problem(
            1,
            "factorial",
            "Compute n! for n >= 0 using iteration.",
            Signature(
                name="factorial",
                args=("n: int",),
                returns="int",
                doc_hint="Compute n! for non-negative n.",
            ),
            g_factorial,
            _cases((0,), (1,), (5,), (8,)),
        )
    )
    add(
        Problem(
            2,
            "gcd",
            "Greatest common divisor using Euclid's algorithm.",
            Signature("gcd", ("a: int", "b: int"), "int", "Greatest common divisor."),
            g_gcd,
            _cases((0, 0), (6, 9), (14, 21), (1071, 462)),
        )
    )
    add(
        Problem(
            3,
            "is_palindrome",
            "Check if a string is a palindrome ignoring case and non-alphanumerics.",
            Signature("is_palindrome", ("s: str",), "bool", "Case-insensitive alnum palindrome."),
            g_is_palindrome,
            _cases(("racecar",), ("A man, a plan, a canal: Panama",), ("ab",), ("",)),
        )
    )
    add(
        Problem(
            4,
            "fibonacci",
            "Return the n-th Fibonacci number (F0=0, F1=1).",
            Signature("fibonacci", ("n: int",), "int", "Return n-th Fibonacci number."),
            g_fibonacci,
            _cases((0,), (1,), (2,), (7,), (12,)),
        )
    )
    add(
        Problem(
            5,
            "reverse_words",
            "Reverse the order of words in a sentence.",
            Signature("reverse_words", ("s: str",), "str", "Reverse word order."),
            g_reverse_words,
            _cases(("hello world",), (" one  two three ",), ("",)),
        )
    )
    add(
        Problem(
            6,
            "run_length_encode",
            "Run-length encode a string as letter-count.",
            Signature("run_length_encode", ("s: str",), "str", "Run-length encode."),
            g_run_length_encode,
            _cases(("",), ("a",), ("aaabbc",), ("hhhhhh",)),
        )
    )
    add(
        Problem(
            7,
            "unique_in_order",
            "Collapse consecutive duplicates in a sequence.",
            Signature("unique_in_order", ("seq: str",), "str", "Remove consecutive duplicates."),
            g_unique_in_order,
            _cases(("",), ("a",), ("aaabcca",), ("ABBCcA",)),
        )
    )
    add(
        Problem(
            8,
            "roman_to_int",
            "Convert a Roman numeral to integer.",
            Signature("roman_to_int", ("s: str",), "int", "Roman numerals to int."),
            g_roman_to_int,
            _cases(("III",), ("IV",), ("LVIII",), ("MCMXCIV",)),
        )
    )
    add(
        Problem(
            9,
            "int_to_roman",
            "Convert an integer (1..3999) to Roman numerals.",
            Signature("int_to_roman", ("n: int",), "str", "Int to Roman numerals."),
            g_int_to_roman,
            _cases((1,), (4,), (58,), (1994,)),
        )
    )
    add(
        Problem(
            10,
            "two_sum_indices",
            "Return indices of two numbers summing to target.",
            Signature(
                "two_sum_indices",
                ("nums: list[int]", "target: int"),
                "tuple[int, int]",
                "Indices i<j s.t. nums[i]+nums[j]==target.",
            ),
            g_two_sum_indices,
            _cases(([2, 7, 11, 15], 9), ([3, 2, 4], 6), ([3, 3], 6)),
        )
    )
    add(
        Problem(
            11,
            "mean",
            "Compute arithmetic mean of a list.",
            Signature("mean", ("nums: list[float]",), "float", "Arithmetic mean (0.0 if empty)."),
            g_mean,
            _cases(([1.0, 2.0, 3.0],), ([],), ([5.0],)),
        )
    )
    add(
        Problem(
            12,
            "flatten",
            "Flatten arbitrarily nested lists/tuples into a flat list.",
            Signature("flatten", ("nested: list",), "list", "Flatten nested lists and tuples."),
            g_flatten,
            _cases(([1, [2, (3,)]],), ([[]],), ([1, [2, [3, 4]], 5],)),
        )
    )
    add(
        Problem(
            13,
            "levenshtein_distance",
            "Compute Levenshtein edit distance between two strings.",
            Signature("levenshtein_distance", ("a: str", "b: str"), "int", "Levenshtein edit distance."),
            g_levenshtein,
            _cases(("kitten", "sitting"), ("", "abc"), ("abc", "abc")),
        )
    )
    add(
        Problem(
            14,
            "is_prime",
            "Primality check for non-negative integers.",
            Signature("is_prime", ("n: int",), "bool", "Return True if n is prime."),
            g_is_prime,
            _cases((0,), (1,), (2,), (17,), (21,), (97,)),
        )
    )

    return tuple(problems)


PROBLEMS: tuple[Problem, ...] = build_problem_bank()


def problem_for_step(step: int, *, seed: int | None = None) -> Problem:
    """Choose a problem deterministically from the bank for a given step.

    The mapping is periodic over the problem count to keep episodes varied while
    stable for a given step. A seed can be supplied to add controlled jitter.
    """

    rng = random.Random(RANDOM_SEED if seed is None else seed)
    # A simple affine transform for extra mixing without statefulness.
    index = (step * 7 + rng.randrange(0, 3)) % len(PROBLEMS)
    return PROBLEMS[index]


def render_user_prompt(p: Problem) -> str:
    """Render the user prompt with natural-language requirement and signature.

    The assistant must return exactly one Python function matching the signature
    and including a concise docstring whose first sentence echoes the hint.
    """

    lines = [
        f"Task: {p.description}",
        "",
        "Write a single Python function that satisfies the task.",
        "Follow the exact signature and include a docstring whose first",
        f"sentence includes: {p.signature.doc_hint}",
        "",
        "Signature:",
        p.signature.header_line(),
    ]
    return "\n".join(lines)


def system_prompt() -> str:
    """Concise system guidance for the policy.

    Rules emphasize deterministic, PEP 8–compliant output and no extra text.
    """

    return (
        "You are an expert Python engineer. Return exactly one complete function "
        "that matches the provided signature. Do not include any explanation, "
        "prefix, suffix, backticks, or tests. The response must begin with 'def ' "
        "and contain only valid Python code. Use clear PEP 8 style, type hints, "
        "and a concise one-line docstring matching the hint. Avoid imports."
    )


__all__ = [
    "RANDOM_SEED",
    "TRAINING_CONFIG",
    "Signature",
    "Problem",
    "PROBLEMS",
    "problem_for_step",
    "render_user_prompt",
    "system_prompt",
]
