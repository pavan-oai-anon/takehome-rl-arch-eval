"""Environment utilities for the '24 game'."""
import random
import re
import xml.etree.ElementTree as ET

# Reproducible randomness
RANDOM_SEED = 42

# Training configuration for ART
TRAINING_CONFIG = {
    "project": "24game",
    "model_name": "agent-24",
    "base_model": "OpenAI/gpt-4",
    "steps": 10,
    "trajectories_per_group": 16,
    "groups_per_step": 1,
    "learning_rate": 1e-5,
    "max_completion_tokens": 64,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_exceptions": 8,
    "cleanup_keep_last": 2,
}

def generate_digits() -> tuple[int, int, int, int]:
    """Generate four random digits between 1 and 9."""
    return tuple(random.randint(1, 9) for _ in range(4))

def parse_solution(xml_str: str) -> str:
    """Extract arithmetic expression from <solution> XML."""
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as exc:
        raise ValueError("Invalid XML") from exc
    if root.tag != "solution" or root.text is None:
        raise ValueError("Invalid solution tag or empty content")
    return root.text.strip()

# Only allow digits, operators, and parentheses
_ALLOWED_EXPR = re.compile(r"^[0-9\+\-\*\/\(\)\s]+$")

def safe_eval(expr: str) -> float:
    """Safely evaluate simple arithmetic expressions."""
    if not _ALLOWED_EXPR.fullmatch(expr):
        raise ValueError("Invalid characters in expression")
    try:
        # Restrict builtins for safety
        result = eval(expr, {"__builtins__": {}}, {})
    except Exception:
        raise ValueError("Error evaluating expression")
    if not isinstance(result, (int, float)):
        raise ValueError("Non-numeric result")
    return float(result)

def evaluate_expression(expr: str) -> float:
    """Compute numeric result of parsed arithmetic expression."""
    return safe_eval(expr)
