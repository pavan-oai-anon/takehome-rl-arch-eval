"""
Environment and configuration for the code snippet classification task.
"""
import random

# Seed for reproducibility
RANDOM_SEED = 42

# Training configuration parameters
TRAINING_CONFIG = {
    "project": "malicious-code-detection",
    "model_name": "code-snippet-classifier",
    "base_model": "Qwen/Qwen2.5-1.5B",
    "steps": 10,
    "trajectories_per_group": 5,
    "groups_per_step": 1,
    "learning_rate": 1e-5,
    "max_completion_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_exceptions": 3,
    "cleanup_keep_last": False,
}

# Snippets for the environment
SNIPPETS = [
    {"language": "Python", "filename": "file1.py", "code": "def hello():\n    print('Hello, world!')", "is_malicious": False},
    {"language": "Python", "filename": "malware.py", "code": "import os\nos.system('rm -rf /')", "is_malicious": True},
    # Add more snippets as needed
]

def get_snippet(index: int) -> dict:
    """Retrieve a code snippet by index."""
    return SNIPPETS[index]

random.seed(RANDOM_SEED)
