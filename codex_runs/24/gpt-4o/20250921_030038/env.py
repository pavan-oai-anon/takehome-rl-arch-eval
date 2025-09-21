"""
Environment helpers for the arithmetic game 24.
"""

RANDOM_SEED = 42

TRAINING_CONFIG = {
    'project': 'arithmetic-game-24',
    'model_name': 'arithmetic_agent_v1',
    'base_model': 'GPT-3',
    'steps': 1000,
    'trajectories_per_group': 10,
    'groups_per_step': 5,
    'learning_rate': 1e-4,
    'max_completion_tokens': 50,
    'temperature': 0.7,
    'top_p': 0.9,
    'max_exceptions': 2,
    'cleanup_keep_last': False,
}

def generate_digits(seed=RANDOM_SEED):
    """
    Generate four random digits between 1 and 9.
    """
    import random
    random.seed(seed)
    return [random.randint(1, 9) for _ in range(4)]

def validate_solution(digits, solution):
    """
    Validate if the solution is correct and achievable with the given digits.
    """
    try:
        # Evaluate the solution string
        from ast import literal_eval
        calculated = literal_eval(solution)
        return calculated == 24
    except (ValueError, SyntaxError):
        return False

