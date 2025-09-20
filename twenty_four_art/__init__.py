"""ART training package for the 24 arithmetic game."""

from .config import TrainingConfig
from .env import Game24, generate_game, render_puzzle
#from .evaluation import evaluate_model
from .rollout import Scenario24, rollout
from .training import run_training, setup_model

__all__ = [
    "TrainingConfig",
    "Game24",
    "generate_game",
    "render_puzzle",
    "Scenario24",
    "rollout",
    "run_training",
    "setup_model",
    #"evaluate_model",
]
