"""ART rollout for the JSON extraction environment.

Implements a single-turn rollout that:
- builds a concise system/user prompt
- queries the model via the ART LocalBackend-compatible API surface
- parses and validates structured output
- logs scalar-only metadata and numeric metrics

Reward shaping mirrors best practices: valid JSON, coverage, types, and
value accuracy (with a smooth term for numeric closeness).
"""
from __future__ import annotations

import requests
import weave
import art
from typing import Any
from openai import AsyncOpenAI

from env import (
    RANDOM_SEED,
    TRAINING_CONFIG,
    SYSTEM_PROMPT,
    get_rng,
    pick_example,
    build_user_prompt,
    extract_json_object,
    compute_reward,
    EXAMPLES,
)


@weave.op
@art.retry(exceptions=(requests.ReadTimeout,))
async def rollout(model: art.Model, step: int, config: dict[str, Any]) -> art.Trajectory:
    """Single-step rollout for JSON extraction.

    Parameters
    - model: ART model handle (assumed LocalBackend for training/inference)
    - step: current training step
    - config: training config dict allowing overrides of generation params
    """
    rng = get_rng(step)
    ex_idx, example = pick_example(step, rng)

    trajectory = art.Trajectory(
        messages_and_choices=[{"role": "system", "content": SYSTEM_PROMPT}],
        metadata={
            "task": "json_extraction",
            "project": str(TRAINING_CONFIG.get("project", "json-extract")),
            "step": int(step),
            "example_index": int(ex_idx),
            "example_id": str(example["id"]),
        },
        reward=0.0,
    )

    user_prompt = build_user_prompt(example)
    trajectory.messages_and_choices.append({"role": "user", "content": user_prompt})

    # LocalBackend-compatible OpenAI transport (as in 2048 example)
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )

    # Allow host config to override generation knobs
    max_tokens = int(config.get("max_completion_tokens", TRAINING_CONFIG["max_completion_tokens"]))
    temperature = float(config.get("temperature", TRAINING_CONFIG["temperature"]))
    top_p = float(config.get("top_p", TRAINING_CONFIG["top_p"]))

    chat_completion = await client.chat.completions.create(
        messages=trajectory.messages(),
        model=model.name,
        max_completion_tokens=max_tokens,
        stream=False,
        temperature=temperature,
        top_p=top_p,
    )

    choice = chat_completion.choices[0]
    content = choice.message.content or ""
    trajectory.messages_and_choices.append(choice)

    payload, parse_err = extract_json_object(content)

    # Defensive validation + reward
    truth = example["label"]
    if payload is None:
        reward, shaped, _ = 0.0, {"valid_json": 0.0, "field_coverage": 0.0, "type_score": 0.0, "value_score": 0.0, "exact_match": 0.0, "total_rel_error": 1.0}, None
        invalid = 1.0
    else:
        reward, shaped, _ = compute_reward(payload, truth)
        invalid = 0.0

    # Record numeric metrics (numbers only)
    trajectory.metrics.update(shaped)
    trajectory.metrics["invalid_solution"] = float(invalid)
    trajectory.metrics["chars_out"] = float(len(content))

    # Scalar-only metadata for aggregation
    trajectory.metadata["parse_error"] = str(parse_err or "none")

    trajectory.reward = float(reward)
    return trajectory

