"""ART rollout for the arithmetic game "24".

Implements a single-turn rollout that:
- samples four digits
- prompts the policy to return <solution>EXPR</solution>
- validates and evaluates the expression safely
- logs scalar metadata and numeric metrics
- computes a shaped reward based on closeness to 24 and rule adherence
"""
from __future__ import annotations

import random
from typing import Any

import art
import requests
import weave
from openai import AsyncOpenAI

from env import (
    RANDOM_SEED,
    TRAINING_CONFIG,
    TARGET_VALUE,
    CLIPPED_ERROR,
    EXACT_SOLUTION_REWARD,
    EPSILON,
    digits_to_string,
    generate_episode_digits,
    new_episode_id,
    numbers_match_episode,
    render_system_prompt,
    render_user_prompt,
    safe_eval_arithmetic,
)


def _smooth_reward(abs_error: float, used_all_digits: bool) -> float:
    """Smooth, bounded reward for near-misses.

    Maps zero error -> 1.0, and errors >= CLIPPED_ERROR -> 0.0. Adds a small
    bonus when the expression uses exactly the given digits to guide learning.
    """

    base = max(0.0, 1.0 - min(abs_error, CLIPPED_ERROR) / CLIPPED_ERROR)
    return base + (0.2 if used_all_digits else 0.0)


@weave.op
@art.retry(exceptions=(requests.ReadTimeout,))
async def rollout(model: art.Model, step: int, config: dict[str, Any]) -> art.Trajectory:
    """Collect one trajectory for the game "24".

    Parameters
    ----------
    model: art.Model
        The model to query for completions.
    step: int
        Current training step; included in metadata for aggregation.
    config: dict[str, Any]
        Training/inference config forwarded from the host loop.
    """

    # Deterministic sampling per step (still varied by episode id/draw).
    random.seed(RANDOM_SEED + step)
    d1, d2, d3, d4 = generate_episode_digits()
    digits_str = digits_to_string((d1, d2, d3, d4))
    episode_id = new_episode_id()

    # System + one user turn (single-step decision problem).
    trajectory = art.Trajectory(
        messages_and_choices=[{"role": "system", "content": render_system_prompt()}],
        metadata={
            "notebook-id": "24",  # scalar string for aggregation
            "project": TRAINING_CONFIG.get("project", "game-24"),
            "step": step,
            "episode_id": episode_id,
            "digits": digits_str,
        },
        reward=0.0,
    )

    user_prompt = render_user_prompt(d1, d2, d3, d4)
    trajectory.messages_and_choices.append({"role": "user", "content": user_prompt})

    client = AsyncOpenAI(base_url=model.inference_base_url, api_key=model.inference_api_key)
    completion = await client.chat.completions.create(
        messages=trajectory.messages(),
        model=model.name,
        max_completion_tokens=int(TRAINING_CONFIG.get("max_completion_tokens", 96)),
        temperature=float(TRAINING_CONFIG.get("temperature", 0.7)),
        top_p=float(TRAINING_CONFIG.get("top_p", 0.9)),
        stream=False,
    )

    choice = completion.choices[0]
    content = choice.message.content
    if not isinstance(content, str):  # pragma: no cover - defensive
        content = str(content)
    trajectory.messages_and_choices.append(choice)

    # Parse <solution> ... </solution>
    xml_error = None
    expr_text: str | None = None
    try:
        # Minimal, robust extraction without bringing XML deps: simple slice
        # while keeping strict tag names; if missing, this will raise.
        start_tag = "<solution>"
        end_tag = "</solution>"
        start = content.index(start_tag) + len(start_tag)
        end = content.index(end_tag, start)
        expr_text = content[start:end].strip()
        if not expr_text:
            xml_error = "empty_solution"
    except Exception:
        xml_error = "missing_or_malformed_xml"

    if xml_error is not None or expr_text is None:
        trajectory.reward = -1.0
        trajectory.metadata["error"] = xml_error or "xml_extract_failed"
        trajectory.metrics["invalid_solution"] = 1.0
        trajectory.metrics["abs_error"] = float("inf")
        trajectory.metrics["used_all_digits"] = 0.0
        trajectory.metrics["is_exact"] = 0.0
        trajectory.metrics["op_count"] = 0.0
        return trajectory

    # Safe evaluation + rule checks
    eval_result = safe_eval_arithmetic(expr_text)
    used_all_digits = numbers_match_episode(eval_result.numbers, (d1, d2, d3, d4))

    if eval_result.error is not None or eval_result.value is None:
        trajectory.reward = -1.0
        trajectory.metadata["error"] = str(eval_result.error)
        trajectory.metrics["invalid_solution"] = 1.0
        trajectory.metrics["abs_error"] = float("inf")
        trajectory.metrics["used_all_digits"] = 1.0 if used_all_digits else 0.0
        trajectory.metrics["is_exact"] = 0.0
        trajectory.metrics["op_count"] = float(eval_result.op_count)
        return trajectory

    # Compute closeness and reward
    abs_error = abs(eval_result.value - TARGET_VALUE)
    is_exact = abs_error <= EPSILON and used_all_digits
    if is_exact:
        reward = EXACT_SOLUTION_REWARD
    else:
        reward = _smooth_reward(abs_error, used_all_digits)

    # Metrics (numbers only) and minimal scalar metadata
    trajectory.reward = float(reward)
    trajectory.metadata["error"] = ""  # keep scalar-only metadata
    trajectory.metrics["abs_error"] = float(abs_error)
    trajectory.metrics["used_all_digits"] = 1.0 if used_all_digits else 0.0
    trajectory.metrics["is_exact"] = 1.0 if is_exact else 0.0
    trajectory.metrics["invalid_solution"] = 0.0
    trajectory.metrics["op_count"] = float(eval_result.op_count)

    return trajectory


