"""Rollout logic for the ART arithmetic 24 environment."""
from __future__ import annotations

from typing import Any

import art
import weave
from openai import AsyncOpenAI

from env import TRAINING_CONFIG, create_puzzle, render_user_prompt, score_solution_xml

SYSTEM_PROMPT = (
    "You solve arithmetic 24 puzzles. Use each provided number exactly once with +, -, *, or /. "
    "Return a single XML block `<solution>...</solution>` containing only the arithmetic expression."
)


@weave.op
@art.retry()
async def rollout(model: art.Model, step: int, config: dict[str, Any]) -> art.Trajectory:
    """Gather a single trajectory for the arithmetic 24 task."""

    runtime_config = {**TRAINING_CONFIG, **(config or {})}
    puzzle = create_puzzle()

    trajectory = art.Trajectory(
        messages_and_choices=[{"role": "system", "content": SYSTEM_PROMPT}],
        metadata={
            "project": runtime_config["project"],
            "step": step,
            "puzzle_id": puzzle.identifier,
            "digits": " ".join(str(d) for d in puzzle.digits),
        },
        reward=0.0,
    )

    trajectory.messages_and_choices.append(
        {"role": "user", "content": render_user_prompt(puzzle)}
    )

    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )

    completion = await client.chat.completions.create(
        model=model.name,
        messages=trajectory.messages(),
        temperature=float(runtime_config["temperature"]),
        top_p=float(runtime_config["top_p"]),
        max_completion_tokens=int(runtime_config["max_completion_tokens"]),
        stream=False,
    )

    choice = completion.choices[0]
    content = choice.message.content or ""
    trajectory.messages_and_choices.append(choice)

    reward, metrics, error_text = score_solution_xml(content, puzzle)
    trajectory.reward = reward

    for name, value in metrics.items():
        trajectory.metrics[name] = float(value)

    trajectory.metadata["invalid_solution_flag"] = float(metrics["invalid_solution"])
    trajectory.metadata["response_length"] = len(content)
    trajectory.metadata["temperature"] = float(runtime_config["temperature"])

    if error_text:
        trajectory.metadata["validation_error"] = error_text

    return trajectory
