"""Utilities for composing Codex prompts targeting OpenPipe ART RL projects."""
from __future__ import annotations

from pathlib import Path


GLOBAL_PROMPT = """You are Codex running with the flags --full-auto --bypass-approvals --bypass-sandbox --trusted-workspace.
Your task is to generate a minimal reinforcement learning task package for OpenPipe's ART framework.
Only create two Python files in the current working directory:
- `env.py` describing the environment helpers and shared utilities.
- `rollout.py` implementing ART rollout logic for that environment.
Follow these rules:
1. Never emit Colab or notebook code (no %magics, widgets, or shell escapes like `!pip`).
2. Use uv for installation snippets when you mention dependencies (e.g. `uv pip install ...`).
3. `env.py` must expose `RANDOM_SEED` (int) and `TRAINING_CONFIG` (dict) providing at least: `project`, `model_name`, `base_model`, `steps`, `trajectories_per_group`, `groups_per_step`, `learning_rate`, `max_completion_tokens`, `temperature`, `top_p`, `max_exceptions`, and `cleanup_keep_last`.
4. Keep hyperparameters and environment constants easy to tweak at the top of the file; prefer small helper functions with docstrings for non-trivial logic.
5. `rollout.py` must import from `env` and define `async def rollout(model: art.Model, step: int, config: dict[str, Any]) -> art.Trajectory` decorated with `@weave.op` and `@art.retry` guards, generating metadata and rewards similar to the 2048 example.
6. Populate trajectory metadata using scalars only (no lists/dicts) so ART aggregation works.
7. Use concise system/user prompts that explain how the policy should format responses.
8. Validate structured outputs defensively and record any validation errors as metadata plus a numeric metric (e.g. `invalid_solution = 1.0`).
9. Assume a LocalBackend for inference/training; add comments when copying memory-tuning values from 2048.py or when GPU selection matters.
10. Avoid defining the training loop or evaluation entry point—the host project supplies a generic `training.py` that will import these files.
11. Prefer type hints, docstrings, and a compact, readable style.
12. Do not create extra files beyond `env.py` and `rollout.py`.
13. Metadata must be a simple scalar value, not a list/dict.
14. Metrics must be a number in trajectory.metrics.
15. You should think deeply about the reward modeling for the task. Rewards are how the agent learns, so you should design them to be as informative as possible. You might want to consider having rewards that are somewhat smooth so that we can actually have some variance to learn.
Make sure to use the codex tools to create the files needed for this - I don't just want example output.
"""


def _load_2048_example() -> str:
    example_path = Path(__file__).with_name("2048.py")
    return example_path.read_text().strip()


EXAMPLE_2048 = _load_2048_example()

USER_PROMPT_24 = (
    "Create an ART reinforcement learning setup for the arithmetic game \"24\". "
    "The environment should present four random digits each episode and the agent must "
    "combine them with +, -, *, or / to reach exactly 24, returning solutions as XML "
    "(e.g. <solution>(3 * (4 + 4))</solution>). Use the shared project scaffolding and "
    "match the ergonomics of the 2048 example."
)


def compose_prompt(user_prompt: str) -> str:
    """Build the final Codex prompt by prepending guidance and the 2048 reference."""

    final_user_prompt = user_prompt.strip()

    sections = [
        GLOBAL_PROMPT,
        "Example Implementation (2048):\n```python\n" + EXAMPLE_2048 + "\n```",
    ]

    if final_user_prompt:
        sections.append("User Prompt:\n" + final_user_prompt)

    return "\n\n".join(sections)


__all__ = ["GLOBAL_PROMPT", "USER_PROMPT_24", "compose_prompt", "EXAMPLE_2048"]
