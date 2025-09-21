"""Generic ART training harness that loads env/rollout modules from a task folder."""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import inspect
import os
import random
import re
import sys
from dataclasses import dataclass, field, fields
from datetime import datetime
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
    backend_path: str | None = None
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
        config.steps = 10
        config.trajectories_per_group = 16
        config.groups_per_step = 8
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


def _resolve_backend_path(config: TrainingConfig) -> Path:
    """Determine where LocalBackend artifacts should live at the project root."""

    root = Path.cwd()
    if config.backend_path:
        path = Path(config.backend_path)
        if not path.is_absolute():
            path = root / path
    else:
        path = root / ".art"
    path.mkdir(parents=True, exist_ok=True)
    return path


async def initialize_backend(config: TrainingConfig, base_task_dir: Path) -> Any:
    """Create a backend that can be shared across multiple models."""

    if config.use_skypilot:
        if SkyPilotBackend is None:
            raise RuntimeError(
                "SkyPilot backend requested but openpipe-art[skypilot] is not installed",
            )

        cluster_kwargs: dict[str, Any] = {"cluster_name": config.skypilot_cluster_name}
        if config.skypilot_art_version:
            cluster_kwargs["art_version"] = config.skypilot_art_version
        if config.skypilot_env_path:
            env_path = Path(config.skypilot_env_path)
            if not env_path.is_absolute():
                env_path = base_task_dir / env_path
            cluster_kwargs["env_path"] = str(env_path)
        if config.skypilot_gpu:
            cluster_kwargs["gpu"] = config.skypilot_gpu

        backend = await SkyPilotBackend.initialize_cluster(**cluster_kwargs)
    else:
        backend_path = _resolve_backend_path(config)
        backend = LocalBackend(in_process=True, path=str(backend_path))

    return backend


_initialized_weave_projects: set[str] = set()


def maybe_enable_weave(project: str) -> bool:
    """Initialize Weave logging once per project when credentials are present."""

    if project in _initialized_weave_projects:
        return True

    if not (os.getenv("WANDB_API_KEY") or os.getenv("WEAVE_API_KEY")):
        return False

    try:
        weave.init(project, settings={"print_call_link": False})
        _initialized_weave_projects.add(project)
        return True
    except Exception as exc:  # pragma: no cover - telemetry optional
        print(f"Weave init failed: {exc}")
        return False


async def setup_model(
    config: TrainingConfig,
    *,
    backend: Any,
    task_dir: Path,
    random_seed: int,
) -> tuple[art.TrainableModel, bool]:
    """Instantiate and register an ART TrainableModel on the shared backend."""

    load_dotenv(task_dir / ".env", override=True)

    random.seed(random_seed)

    sanitized_name = re.sub(r"[^A-Za-z0-9_-]+", "_", config.model_name)
    if not sanitized_name:
        sanitized_name = f"model-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    config.model_name = sanitized_name

    model = art.TrainableModel(
        name=config.model_name,
        project=config.project,
        base_model=config.base_model,
    )
    model._internal_config = art.dev.InternalModelConfig(  # type: ignore[attr-defined]
        init_args=art.dev.InitArgs(max_seq_length=8192),
        engine_args=art.dev.EngineArgs(
            enforce_eager=True,
            gpu_memory_utilization=0.4,
        ),
    )

    await model.register(backend)

    weave_enabled = maybe_enable_weave(config.project)

    return model, weave_enabled


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
                    
        print(f"Training step {step} finished")

    print(f"Training finished at step {await model.get_step()}")


@dataclass
class TaskRuntime:
    """Container describing a single task to train."""

    name: str
    task_dir: Path
    env_module: Any
    rollout_module: Any
    rollout_fn: TrajectoryCallable
    config: TrainingConfig
    rollout_config: dict[str, Any]
    random_seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one or more ART agents using generated task folders")
    parser.add_argument(
        "task_dirs",
        type=Path,
        nargs="+",
        help="One or more directories containing env.py and rollout.py produced by Codex",
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
        "--backend-path",
        dest="backend_path",
        help="Directory to store shared backend artifacts when running locally",
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


def apply_overrides(config: TrainingConfig, args: argparse.Namespace) -> None:
    """Apply CLI overrides to a training config in-place."""

    if args.steps is not None:
        config.steps = args.steps
    if args.trajectories_per_group is not None:
        config.trajectories_per_group = args.trajectories_per_group
    if args.groups_per_step is not None:
        config.groups_per_step = args.groups_per_step
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.temperature is not None:
        config.temperature = args.temperature
    if args.top_p is not None:
        config.top_p = args.top_p
    if args.backend_path is not None:
        config.backend_path = args.backend_path
    if args.use_skypilot:
        config.use_skypilot = True
    if args.skypilot_cluster_name is not None:
        config.skypilot_cluster_name = args.skypilot_cluster_name
    if args.skypilot_art_version is not None:
        config.skypilot_art_version = args.skypilot_art_version
    if args.skypilot_env_path is not None:
        config.skypilot_env_path = args.skypilot_env_path
    if args.skypilot_gpu is not None:
        config.skypilot_gpu = args.skypilot_gpu
    if args.teardown_remote_backend:
        config.teardown_remote_backend = True


def ensure_backend_compat(base: TrainingConfig, candidate: TrainingConfig, task_name: str) -> None:
    """Validate that two configs can safely share the same backend."""

    shared_fields = [
        "use_skypilot",
        "skypilot_cluster_name",
        "skypilot_art_version",
        "skypilot_env_path",
        "skypilot_gpu",
        "backend_path",
    ]

    for field_name in shared_fields:
        if getattr(base, field_name) != getattr(candidate, field_name):
            raise ValueError(
                f"Task '{task_name}' has incompatible '{field_name}' for a shared backend",
            )


async def main() -> None:
    load_dotenv()
    args = parse_args()

    tasks: list[TaskRuntime] = []

    for raw_dir in args.task_dirs:
        task_dir = raw_dir.resolve()
        if not task_dir.exists():
            raise FileNotFoundError(f"Task directory {task_dir} does not exist")

        env_module, rollout_module = load_task_modules(task_dir)
        training_config, rollout_config = resolve_training_config(env_module)
        apply_overrides(training_config, args)

        try:
            relative_path = task_dir.relative_to(Path.cwd())
        except ValueError:
            relative_path = task_dir
        relevant_parts = relative_path.parts[-3:] if len(relative_path.parts) >= 3 else relative_path.parts
        project_component = relevant_parts[0] if relevant_parts else task_dir.name
        project_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", project_component)
        if not project_slug:
            project_slug = "project"
        training_config.project = project_slug

        rollout_config = {**training_config.as_dict(), **training_config.extra}

        rollout_fn = getattr(rollout_module, "rollout", None)
        if rollout_fn is None or not inspect.iscoroutinefunction(rollout_fn):
            raise AttributeError(
                f"rollout.py in '{task_dir}' must define an async function named 'rollout'",
            )

        random_seed = int(getattr(env_module, "RANDOM_SEED", 1234))

        tasks.append(
            TaskRuntime(
                name=task_dir.name,
                task_dir=task_dir,
                env_module=env_module,
                rollout_module=rollout_module,
                rollout_fn=rollout_fn,  # type: ignore[arg-type]
                config=training_config,
                rollout_config=rollout_config,
                random_seed=random_seed,
            )
        )

    if not tasks:
        raise RuntimeError("No tasks provided for training")

    base_config = tasks[0].config
    teardown_remote = base_config.teardown_remote_backend

    for task in tasks[1:]:
        ensure_backend_compat(base_config, task.config, task.name)
        teardown_remote = teardown_remote or task.config.teardown_remote_backend

    base_config.teardown_remote_backend = teardown_remote

    backend = await initialize_backend(base_config, tasks[0].task_dir)

    try:
        for task in tasks:
            sys.modules["env"] = task.env_module
            sys.modules["rollout"] = task.rollout_module

            try:
                relative_path = task.task_dir.relative_to(Path.cwd())
            except ValueError:
                relative_path = task.task_dir
            relevant_parts = relative_path.parts[-3:] if len(relative_path.parts) >= 3 else relative_path.parts
            raw_name = "-".join(relevant_parts)
            sanitized_name = re.sub(r"[^A-Za-z0-9_-]+", "_", raw_name)
            if not sanitized_name:
                sanitized_name = datetime.now().strftime("run-%Y%m%d-%H%M%S")

            task.config.model_name = sanitized_name

            model, weave_enabled = await setup_model(
                task.config,
                backend=backend,
                task_dir=task.task_dir,
                random_seed=task.random_seed,
            )

            task.rollout_config["model_name"] = task.config.model_name

            await run_training(
                model,
                task.config,
                task.rollout_fn,
                task.rollout_config,
                weave_enabled=weave_enabled,
            )
    finally:
        if base_config.use_skypilot and base_config.teardown_remote_backend:
            down = getattr(backend, "down", None)
            if callable(down):
                maybe_coro = down()
                if inspect.isawaitable(maybe_coro):
                    await maybe_coro


if __name__ == "__main__":
    asyncio.run(main())
