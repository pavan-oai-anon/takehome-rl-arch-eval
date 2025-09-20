"""Reusable configuration dataclasses for the 24 game trainer."""
from __future__ import annotations

from dataclasses import dataclass

import art


@dataclass
class TrainingConfig:
    """Hyperparameters and runtime knobs for training."""

    project: str = "game-24"
    model_name: str = "agent-24"
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    steps: int = 1000
    trajectories_per_group: int = 64
    groups_per_step: int = 2
    learning_rate: float = 1e-5
    max_completion_tokens: int = 192
    temperature: float = 0.7
    top_p: float = 0.95
    max_exceptions: int = 8
    cleanup_keep_last: int = 1

    def art_train_config(self) -> art.TrainConfig:
        """Translate the dataclass into an ART TrainConfig."""

        return art.TrainConfig(learning_rate=self.learning_rate)
