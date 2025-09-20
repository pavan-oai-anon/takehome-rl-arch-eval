"""Environment helpers for JSON extraction task."""
import random
from typing import TypedDict, Dict, List

RANDOM_SEED: int = 1234

TRAINING_CONFIG: Dict[str, object] = {
    "project": "json-extraction",
    "model_name": "json-extractor",
    "base_model": "gpt-3.5-turbo",
    "steps": 100,
    "trajectories_per_group": 5,
    "groups_per_step": 1,
    "learning_rate": 1e-5,
    "max_completion_tokens": 256,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_exceptions": 2,
    "cleanup_keep_last": 2,
}

class Example(TypedDict):
    """A single data example with unstructured text and metadata."""
    text: str
    metadata: Dict[str, str]

# Predefined examples for the environment
_EXAMPLES: List[Example] = [
    {"text": "Name: Alice\nAge: 30\nLocation: Seattle", "metadata": {"id": "1"}},
    {"text": "Bob, 45, works in New York.", "metadata": {"id": "2"}},
    {"text": "Employee: Carol | Age=28 | City=Boston", "metadata": {"id": "3"}},
    {"text": "Derek is 52 years old and lives in Chicago.", "metadata": {"id": "4"}},
    {"text": "Eva; age 37; based out of San Francisco", "metadata": {"id": "5"}},
    {"text": "Frank (29) — Location: Miami", "metadata": {"id": "6"}},
    {"text": "Gina | 40 yrs | Dallas", "metadata": {"id": "7"}},
    {"text": "Helen: age thirty-two; city: Austin", "metadata": {"id": "8"}},
    {"text": "Ian - 22 - Denver", "metadata": {"id": "9"}},
    {"text": "Name=Jack; Age=34; Location=Houston", "metadata": {"id": "10"}},
]

def sample_example() -> Example:
    """Return a random example from the predefined set."""
    return random.choice(_EXAMPLES)
