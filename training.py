"""Generic ART training harness that loads env/rollout modules from a task folder."""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import inspect
import os
import random
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Awaitable, Callable

import art
from art.local import LocalBackend
from dotenv import load_dotenv
import weave

try:  # optional dependency for remote provisioning
    from art.skypilot import SkyPilotBackend
except ImportError:  # pragma: no cover - optional extra
    SkyPilotBackend = None  # type: ignore[assignment]


TrajectoryCallable = Callable[[art.Model, int, dict[str, Any]], Awaitable[art.Trajectory]]


@dataclass
class TrainingConfig:
    """Runtime knobs consumed by the generic training loop."""

    project: str = "art-task"
    model_name: str = "art-agent"
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    steps: int = 10
    trajectories_per_group: int = 8
    groups_per_step: int = 2
    learning_rate: float = 1e-5
    max_completion_tokens: int = 192
    temperature: float = 0.7
    top_p: float = 0.95
    max_exceptions: int = 8
    cleanup_keep_last: int = 1
    use_skypilot: bool = False
    skypilot_cluster_name: str = "art-task-cluster"
    skypilot_art_version: str | None = None
    skypilot_env_path: str | None = None
    skypilot_gpu: str | None = None
    teardown_remote_backend: bool = False
    extra: dict[str, Any] = field(default_factory=dict, init=False)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "TrainingConfig":
        """Merge user-provided config with defaults, preserving unknown keys."""

        known = {f.name for f in fields(cls) if f.init}
        init_kwargs = {key: value for key, value in mapping.items() if key in known}
        config = cls(**init_kwargs)
        config.extra = {key: value for key, value in mapping.items() if key not in known}
        # override the base model with ours and training steps, trajectories per group, groups per step
        config.base_model = "Qwen/Qwen2.5-1.5B-Instruct"
        config.steps = 100
        config.trajectories_per_group = 64
        config.groups_per_step = 2
        return config

    def as_dict(self) -> dict[str, Any]:
        """Serialize init-visible fields for downstream consumption."""

        data = {f.name: getattr(self, f.name) for f in fields(self) if f.init}
        return data


def import_module(path: Path, module_name: str, alias: str | None = None) -> Any:
    """Load a Python module from an arbitrary file path."""

    if not path.exists():
        raise FileNotFoundError(f"Expected module at {path}")

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for module at {path}")

    module = importlib.util.module_from_spec(spec)
    if alias:
        sys.modules[alias] = module
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


async def setup_model(
    config: TrainingConfig,
    task_dir: Path,
) -> tuple[art.TrainableModel, Any, bool]:
    """Instantiate and register an ART TrainableModel using task-local storage."""

    load_dotenv(task_dir / ".env")
    load_dotenv()  # allow repository-level overrides

    random.seed(getattr(sys.modules.get("env"), "RANDOM_SEED", 1234))

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

    backend_path = task_dir / ".art"
    backend_path.mkdir(parents=True, exist_ok=True)

    if config.use_skypilot:
        if SkyPilotBackend is None:
            raise RuntimeError(
                "SkyPilot backend requested but openpipe-art[skypilot] is not installed",
            )

        cluster_kwargs: dict[str, Any] = {"cluster_name": config.skypilot_cluster_name}
        if config.skypilot_art_version:
            cluster_kwargs["art_version"] = config.skypilot_art_version
        if config.skypilot_env_path:
            cluster_kwargs["env_path"] = str(task_dir / config.skypilot_env_path)
        if config.skypilot_gpu:
            cluster_kwargs["gpu"] = config.skypilot_gpu

        backend = await SkyPilotBackend.initialize_cluster(**cluster_kwargs)
    else:
        backend = LocalBackend(in_process=True, path=str(backend_path))

    await model.register(backend)

    weave_enabled = False
    if os.getenv("WANDB_API_KEY") or os.getenv("WEAVE_API_KEY"):
        try:
            weave.init(config.project, settings={"print_call_link": False})
            weave_enabled = True
        except Exception as exc:  # pragma: no cover - telemetry optional
            print(f"Weave init failed: {exc}")

    return model, backend, weave_enabled


async def run_training(
    model: art.TrainableModel,
    config: TrainingConfig,
    rollout_fn: TrajectoryCallable,
    rollout_config: dict[str, Any],
    *,
    weave_enabled: bool = False,
) -> None:
    """Run the gather/train loop for the provided rollout function."""

    start_step = await model.get_step()

    for step in range(start_step, config.steps):
        trajectory_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout_fn(model, step, rollout_config)
                    for _ in range(config.trajectories_per_group)
                )
                for _ in range(config.groups_per_step)
            ),
            pbar_desc=f"gather step {step}",
            max_exceptions=config.max_exceptions,
        )

        try:
            await model.delete_checkpoints(keep_last=config.cleanup_keep_last)
        except TypeError:  # pragma: no cover - older ART versions
            await model.delete_checkpoints()

        await model.train(
            trajectory_groups,
            config=art.TrainConfig(learning_rate=config.learning_rate),
            _config={"logprob_calculation_chunk_size": 8},
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
                    weave.log({"step": step, "mean_reward": sum(rewards) / len(rewards)})
                except Exception as exc:  # pragma: no cover - telemetry optional
                    print(f"Weave logging failed: {exc}")

    print(f"Training finished at step {await model.get_step()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ART agent using a generated task folder")
    parser.add_argument(
        "task_dir",
        type=Path,
        help="Directory containing env.py and rollout.py produced by Codex",
    )
    parser.add_argument("--steps", type=int, help="Override the number of training steps")
    parser.add_argument(
        "--trajectories-per-group",
        type=int,
        dest="trajectories_per_group",
        help="Override rollouts requested per trajectory group",
    )
    parser.add_argument(
        "--groups-per-step",
        type=int,
        dest="groups_per_step",
        help="Override the number of trajectory groups gathered per step",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        dest="learning_rate",
        help="Override the ART learning rate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Override sampling temperature for rollouts",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="Override nucleus sampling parameter for rollouts",
    )
    parser.add_argument(
        "--use-skypilot",
        action="store_true",
        help="Provision a SkyPilot cluster instead of running locally",
    )
    parser.add_argument(
        "--cluster-name",
        dest="skypilot_cluster_name",
        help="Name to register the SkyPilot cluster under",
    )
    parser.add_argument(
        "--skypilot-art-version",
        dest="skypilot_art_version",
        help="openpipe-art version to install remotely (defaults to client version)",
    )
    parser.add_argument(
        "--skypilot-env-path",
        dest="skypilot_env_path",
        help="Path within the task folder to an env file to upload",
    )
    parser.add_argument(
        "--skypilot-gpu",
        dest="skypilot_gpu",
        help="GPU type to request for the SkyPilot cluster (e.g. H100)",
    )
    parser.add_argument(
        "--tear-down-remote",
        action="store_true",
        dest="teardown_remote_backend",
        help="Shut down the SkyPilot cluster after training",
    )
    return parser.parse_args()


def load_task_modules(task_dir: Path) -> tuple[Any, Any]:
    """Import env.py and rollout.py from the provided task directory."""

    env_module = import_module(task_dir / "env.py", f"task_env_{task_dir.name}", alias="env")
    rollout_module = import_module(
        task_dir / "rollout.py", f"task_rollout_{task_dir.name}", alias="rollout"
    )
    return env_module, rollout_module


def resolve_training_config(env_module: Any) -> tuple[TrainingConfig, dict[str, Any]]:
    """Build the runtime TrainingConfig and expose the rollouts' config dict."""

    raw_config = getattr(env_module, "TRAINING_CONFIG", {})
    if not isinstance(raw_config, dict):
        raise TypeError("TRAINING_CONFIG must be a dict")

    training_config = TrainingConfig.from_mapping(raw_config)
    rollout_config = {**training_config.as_dict(), **training_config.extra}
    return training_config, rollout_config


async def main() -> None:
    args = parse_args()
    task_dir = args.task_dir.resolve()
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory {task_dir} does not exist")

    env_module, rollout_module = load_task_modules(task_dir)
    training_config, rollout_config = resolve_training_config(env_module)

    if args.steps is not None:
        training_config.steps = args.steps
    if args.trajectories_per_group is not None:
        training_config.trajectories_per_group = args.trajectories_per_group
    if args.groups_per_step is not None:
        training_config.groups_per_step = args.groups_per_step
    if args.learning_rate is not None:
        training_config.learning_rate = args.learning_rate
    if args.temperature is not None:
        training_config.temperature = args.temperature
        rollout_config["temperature"] = args.temperature
    if args.top_p is not None:
        training_config.top_p = args.top_p
        rollout_config["top_p"] = args.top_p
    if args.use_skypilot:
        training_config.use_skypilot = True
    if args.skypilot_cluster_name is not None:
        training_config.skypilot_cluster_name = args.skypilot_cluster_name
    if args.skypilot_art_version is not None:
        training_config.skypilot_art_version = args.skypilot_art_version
    if args.skypilot_env_path is not None:
        training_config.skypilot_env_path = args.skypilot_env_path
    if args.skypilot_gpu is not None:
        training_config.skypilot_gpu = args.skypilot_gpu
    if args.teardown_remote_backend:
        training_config.teardown_remote_backend = True

    rollout_fn = getattr(rollout_module, "rollout", None)
    if rollout_fn is None or not inspect.iscoroutinefunction(rollout_fn):
        raise AttributeError("rollout.py must define an async function named 'rollout'")

    model, backend, weave_enabled = await setup_model(training_config, task_dir)

    try:
        await run_training(
            model,
            training_config,
            rollout_fn,  # type: ignore[arg-type]
            rollout_config,
            weave_enabled=weave_enabled,
        )
    finally:
        if training_config.use_skypilot and training_config.teardown_remote_backend:
            down = getattr(backend, "down", None)
            if callable(down):
                maybe_coro = down()
                if inspect.isawaitable(maybe_coro):
                    await maybe_coro


if __name__ == "__main__":
    asyncio.run(main())
