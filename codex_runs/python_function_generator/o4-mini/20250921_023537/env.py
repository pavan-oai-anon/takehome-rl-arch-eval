"""Environment helpers for Python function generation RL task."""

from typing import Any, Dict, List, Tuple, TypedDict
import random

# Random seed for reproducibility
RANDOM_SEED: int = 42

# Training configuration for ART
TRAINING_CONFIG: Dict[str, Any] = {
    "project": "function-generation",
    "model_name": "func-gen-agent",
    "base_model": "gpt-4o",
    "steps": 100,
    "trajectories_per_group": 8,
    "groups_per_step": 1,
    "learning_rate": 1e-4,
    "max_completion_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_exceptions": 2,
    "cleanup_keep_last": 1,
}


class Scenario(TypedDict):
    """Specification for a single function generation scenario."""
    id: int
    description: str
    function_name: str
    arg_names: List[str]
    docstring: str
    golden: str
    test_cases: List[Tuple[List[Any], Any]]

# Seeded scenarios with natural-language requirements, signatures, and golden implementations
SCENARIOS: List[Scenario] = [
    {
        "id": 0,
        "description": "Return the sum of all numbers in a list.",
        "function_name": "sum_list",
        "arg_names": ["lst"],
        "docstring": '"""Return the sum of all numbers in the list."""',
        "golden": '''def sum_list(lst):
    """Return the sum of all numbers in the list."""
    return sum(lst)
''',
        "test_cases": [(([1, 2, 3],), 6), (([],), 0), (([0, -1, 1],), 0)],
    },
    {
        "id": 1,
        "description": "Compute the factorial of a non-negative integer.",
        "function_name": "factorial",
        "arg_names": ["n"],
        "docstring": '"""Compute the factorial of n."""',
        "golden": '''def factorial(n):
    """Compute the factorial of n."""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
''',
        "test_cases": [((0,), 1), ((1,), 1), ((5,), 120)],
    },
    {
        "id": 2,
        "description": "Return the nth Fibonacci number (0-indexed).",
        "function_name": "fib",
        "arg_names": ["n"],
        "docstring": '"""Return the nth Fibonacci number."""',
        "golden": '''def fib(n):
    """Return the nth Fibonacci number."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
''',
        "test_cases": [((0,), 0), ((1,), 1), ((7,), 13)],
    },
    {
        "id": 3,
        "description": "Reverse the given string.",
        "function_name": "reverse_string",
        "arg_names": ["s"],
        "docstring": '"""Return the reverse of the string."""',
        "golden": '''def reverse_string(s):
    """Return the reverse of the string."""
    return s[::-1]
''',
        "test_cases": [(('abc',), 'cba'), (('',), '')],
    },
    {
        "id": 4,
        "description": "Determine if a number is prime.",
        "function_name": "is_prime",
        "arg_names": ["n"],
        "docstring": '"""Return True if n is prime, else False."""',
        "golden": '''def is_prime(n):
    """Return True if n is prime, else False."""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
''',
        "test_cases": [((2,), True), ((4,), False), ((17,), True)],
    },
    {
        "id": 5,
        "description": "Compute the greatest common divisor of two integers.",
        "function_name": "gcd",
        "arg_names": ["a", "b"],
        "docstring": '"""Return the GCD of a and b."""',
        "golden": '''def gcd(a, b):
    """Return the GCD of a and b."""
    while b:
        a, b = b, a % b
    return a
''',
        "test_cases": [((48, 18), 6), ((7, 3), 1)],
    },
    {
        "id": 6,
        "description": "Flatten a list of lists into a single list.",
        "function_name": "flatten",
        "arg_names": ["lst"],
        "docstring": '"""Flatten one level of nesting in lst."""',
        "golden": '''def flatten(lst):
    """Flatten one level of nesting in lst."""
    result = []
    for sub in lst:
        result.extend(sub)
    return result
''',
        "test_cases": [(([[1, 2], [3]],), [1, 2, 3]), (([],), [])],
    },
    {
        "id": 7,
        "description": "Find the maximum value in a list.",
        "function_name": "max_in_list",
        "arg_names": ["lst"],
        "docstring": '"""Return the maximum element in lst."""',
        "golden": '''def max_in_list(lst):
    """Return the maximum element in lst."""
    if not lst:
        return None
    return max(lst)
''',
        "test_cases": [(([1, 3, 2],), 3), (([],), None)],
    },
    {
        "id": 8,
        "description": "Merge two sorted lists into one sorted list.",
        "function_name": "merge_sorted",
        "arg_names": ["a", "b"],
        "docstring": '"""Merge two sorted lists a and b."""',
        "golden": '''def merge_sorted(a, b):
    """Merge two sorted lists a and b."""
    i = j = 0
    result = []
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            result.append(a[i]); i += 1
        else:
            result.append(b[j]); j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result
''',
        "test_cases": [(([1, 3], [2, 4]), [1, 2, 3, 4]), (([], []), [])],
    },
    {
        "id": 9,
        "description": "Count the number of words in a string.",
        "function_name": "count_words",
        "arg_names": ["s"],
        "docstring": '"""Return the number of words in s."""',
        "golden": '''def count_words(s):
    """Return the number of words in s."""
    return len(s.split())
''',
        "test_cases": [(('hello world',), 2), (('',), 0)],
    },
]


def get_scenario(step: int) -> Scenario:
    """Select a scenario based on the step index deterministically."""
    return SCENARIOS[step % len(SCENARIOS)]
