"""Utilities for composing Codex prompts targeting OpenPipe ART RL projects."""
from __future__ import annotations

from pathlib import Path


GLOBAL_PROMPT = """You are Codex running with the flags --full-auto --bypass-approvals --bypass-sandbox --trusted-workspace.
Your task is to generate a complete, runnable reinforcement learning project that uses OpenPipe's ART framework.
Follow these requirements:
1. Never emit Colab or notebook-specific code (no %magics, no display widgets, no shell escapes like `!pip`).
2. Use uv for every dependency or tooling operation (e.g. `uv pip install openpipe-art[backend]==0.4.11 --prerelease allow`).
3. Target Python scripts with clear module boundaries, mirroring the structure shown in 2048.py: environment definition, rollout logic, training loop, and post-training evaluation.
4. Define and register an `art.TrainableModel` with a `LocalBackend` (use `in_process=True` and persist checkpoints under `./.art`).
5. Implement a rollout function decorated with `@weave.op` that yields `art.Trajectory` objects, capturing metadata and reward shaping similar to the 2048 example.
6. Create an asyncio-driven training entry point (e.g. `async def main()` and `if __name__ == "__main__": asyncio.run(main())`).
7. Include a gather/train loop that deletes older checkpoints, trains on grouped trajectories, and logs meaningful metrics to Weave/W&B when credentials are available.
8. Provide a lightweight evaluation routine that reloads the freshly trained LoRA weights (via `FastLanguageModel.from_pretrained`) and demonstrates one or two inference rollouts in the target environment.
9. Keep prompts and system messages concise but explicit about action formats the policy must return.
10. Prefer pure Python helpers, type hints, and small docstrings to describe non-obvious logic.
11. Make it easy to adapt hyperparameters, number of trajectories, and reward shaping constants.
12. Validate XML or structured actions defensively; surface invalid moves with informative errors.
13. Assume local execution on a single GPU; include comments when a config is a memory-saving tweak copied from 2048.py.
14. Output shell snippets for uv installation or job kicks only when necessary; never assume Colab hardware.
15. Organize files so the main training job can be launched with a single uv-invoked python command.
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
