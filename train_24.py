"""Async training entry point for the 24 arithmetic game."""
from __future__ import annotations

import argparse
import asyncio

from twenty_four_art import (
    TrainingConfig,
    evaluate_model,
    run_training,
    setup_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ART agent for the 24 game")
    parser.add_argument("--steps", type=int, help="Number of training steps to run")
    parser.add_argument(
        "--trajectories-per-group",
        type=int,
        dest="trajectories_per_group",
        help="How many rollouts to request per trajectory group",
    )
    parser.add_argument(
        "--groups-per-step",
        type=int,
        dest="groups_per_step",
        help="How many trajectory groups to gather per training step",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        dest="learning_rate",
        help="Learning rate for ART fine-tuning",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="How many evaluation rollouts to print after training",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    config = TrainingConfig()
    if args.steps is not None:
        config.steps = args.steps
    if args.trajectories_per_group is not None:
        config.trajectories_per_group = args.trajectories_per_group
    if args.groups_per_step is not None:
        config.groups_per_step = args.groups_per_step
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate

    model, _backend, weave_enabled = await setup_model(config)
    await run_training(model, config, weave_enabled=weave_enabled)
    await evaluate_model(model, config, samples=args.samples)


if __name__ == "__main__":
    asyncio.run(main())
