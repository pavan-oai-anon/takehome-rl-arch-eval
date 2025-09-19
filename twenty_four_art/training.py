"""Training orchestration for the 24 ART project."""
from __future__ import annotations

import os
import random
from statistics import mean

import art
from art.local import LocalBackend
import weave
from dotenv import load_dotenv

from .config import TrainingConfig
from .env import RANDOM_SEED
from .rollout import Scenario24, rollout


async def setup_model(config: TrainingConfig) -> tuple[art.TrainableModel, LocalBackend, bool]:
    """Instantiate and register an ART TrainableModel."""

    load_dotenv()
    random.seed(RANDOM_SEED)

    model = art.TrainableModel(
        name=config.model_name,
        project=config.project,
        base_model=config.base_model,
    )
    model._internal_config = art.dev.InternalModelConfig(  # type: ignore[attr-defined]
        init_args=art.dev.InitArgs(max_seq_length=8192),
        engine_args=art.dev.EngineArgs(
            enforce_eager=True,
            gpu_memory_utilization=0.8,
        ),
    )

    backend = LocalBackend(in_process=True, path="./.art")
    await model.register(backend)

    weave_enabled = False
    if os.getenv("WANDB_API_KEY") or os.getenv("WEAVE_API_KEY"):
        try:
            weave.init(config.project, settings={"print_call_link": False})
            weave_enabled = True
        except Exception as exc:  # pragma: no cover - telemetry is optional
            print(f"Weave init failed: {exc}")

    return model, backend, weave_enabled


async def run_training(
    model: art.TrainableModel,
    config: TrainingConfig,
    *,
    weave_enabled: bool = False,
) -> None:
    """Gather trajectories, train, and manage checkpoints."""

    start_step = await model.get_step()

    for step in range(start_step, config.steps):
        trajectory_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, Scenario24(step=step, config=config))
                    for _ in range(config.trajectories_per_group)
                )
                for _ in range(config.groups_per_step)
            ),
            pbar_desc=f"gather step {step}",
            max_exceptions=config.max_exceptions,
        )

        try:
            await model.delete_checkpoints(keep_last=config.cleanup_keep_last)
        except TypeError:
            await model.delete_checkpoints()

        await model.train(
            trajectory_groups,
            config=config.art_train_config(),
            _config={"logprob_calculation_chunk_size": 8},  # mirrors memory tweak in 2048 example
        )

        if weave_enabled:
            rewards: list[float] = []
            for group in trajectory_groups:
                for trajectory in getattr(group, "trajectories", []):
                    reward = getattr(trajectory, "reward", None)
                    if reward is not None:
                        rewards.append(float(reward))
            if rewards:
                try:
                    weave.log({"step": step, "mean_reward": mean(rewards)})
                except Exception as exc:  # pragma: no cover - telemetry optional
                    print(f"Weave logging failed: {exc}")

    print(f"Training finished at step {await model.get_step()}")
