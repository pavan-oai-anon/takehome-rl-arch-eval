"""ART rollout for the malicious code classification environment.

The rollout presents one snippet per episode and expects a strict JSON reply:
{"is_malicious": <bool>, "explanation": "<short>"}

We validate structure defensively and shape rewards to encourage:
- correct classification
- valid JSON with exactly the required keys
- concise explanations (<=160 chars, single paragraph)

Assumes a LocalBackend for inference/training.
"""
from __future__ import annotations

import json
import re
from typing import Any

import art
import requests
import weave
from openai import AsyncOpenAI

from env import (
    RANDOM_SEED,
    TRAINING_CONFIG,
    select_example_for_step,
    system_prompt,
    user_prompt,
    safe_truncate,
)


def _first_json_obj(text: str) -> tuple[dict[str, Any] | None, str]:
    """Extract and parse the first JSON object in text, if any.

    Returns a tuple (obj, error_message). On success, error_message is "".
    """

    # Fast path: whole string is JSON.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, ""
    except Exception as exc:  # fall through to regex
        last_err = str(exc)
    else:
        last_err = "not a dict"

    # Try to find a {...} region (non-greedy), then parse.
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        snippet = match.group(0)
        try:
            obj2 = json.loads(snippet)
            if isinstance(obj2, dict):
                return obj2, ""
            return None, "json-is-not-object"
        except Exception as exc:  # pragma: no cover - defensive
            return None, safe_truncate(f"json-parse-error: {exc}", 180)
    return None, safe_truncate(f"no-json-found: {last_err}", 180)


def _bool_like(value: Any) -> tuple[bool | None, bool]:
    """Coerce a JSON field to bool if it's clearly boolean-like.

    Returns (coerced_bool_or_None, used_coercion_flag).
    """

    if isinstance(value, bool):
        return value, False
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true", True
    return None, False


@weave.op
@art.retry(exceptions=(requests.ReadTimeout,))
async def rollout(model: art.Model, step: int, config: dict[str, Any]) -> art.Trajectory:
    """Gather one labeled trajectory for a single code snippet.

    Parameters
    - model: ART model wrapper (LocalBackend assumed by host).
    - step: training step index; selects the episode example deterministically.
    - config: training/inference knobs; falls back to TRAINING_CONFIG values.
    """

    example = select_example_for_step(step)

    sys_msg = system_prompt()
    usr_msg = user_prompt(example.language, example.filename, example.code)

    traj = art.Trajectory(
        messages_and_choices=[{"role": "system", "content": sys_msg}],
        metadata={
            # Scalars only so ART aggregation works across runs.
            "step": int(step),
            "example_id": int(example.id),
            "language": example.language,
            "filename": example.filename,
            "label": int(1 if example.label else 0),
            "model_name": getattr(model, "name", "unknown"),
            # Filled after validation
            "parse_error": "",
            "predicted": -1,  # -1 unknown, 0 benign, 1 malicious
            "explanation_len": 0,
        },
        reward=0.0,
    )

    traj.messages_and_choices.append({"role": "user", "content": usr_msg})

    # Inference call via OpenAI-compatible client.
    client = AsyncOpenAI(base_url=model.inference_base_url, api_key=model.inference_api_key)

    max_tokens = int(config.get("max_completion_tokens", TRAINING_CONFIG["max_completion_tokens"]))
    temperature = float(config.get("temperature", TRAINING_CONFIG["temperature"]))
    top_p = float(config.get("top_p", TRAINING_CONFIG["top_p"]))

    chat = await client.chat.completions.create(
        model=model.name,
        messages=traj.messages(),
        max_completion_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=False,
    )

    choice = chat.choices[0]
    content = choice.message.content or ""
    traj.messages_and_choices.append(choice)

    # ----------------------
    # Validate and score
    # ----------------------
    json_obj, parse_err = _first_json_obj(content)

    # Metrics (numeric) and shaping components
    json_valid = 0.0
    invalid_solution = 0.0
    correctness = 0.0
    structure_score = 0.0
    style_score = 0.0

    predicted = -1
    explanation_len = 0
    extra_keys = 0.0

    if json_obj is None:
        invalid_solution = 1.0
        structure_score -= 0.4  # penalize missing/invalid JSON
        traj.metadata["parse_error"] = parse_err
    else:
        json_valid = 1.0
        # Required fields
        keys = set(json_obj.keys())
        extra = keys - {"is_malicious", "explanation"}
        extra_keys = float(len(extra))
        if extra_keys:
            # Mild penalty to encourage exact schema
            structure_score -= min(0.05 * extra_keys, 0.2)

        # is_malicious
        coerced, used_coercion = _bool_like(json_obj.get("is_malicious"))
        if coerced is None:
            invalid_solution = 1.0
            structure_score -= 0.3
            traj.metadata["parse_error"] = "is_malicious not boolean-like"
        else:
            predicted = 1 if coerced else 0
            if used_coercion:
                # Soft penalty for requiring coercion
                structure_score -= 0.05

        # explanation
        explanation = json_obj.get("explanation")
        if isinstance(explanation, str):
            explanation_len = len(explanation)
            # brevity incentive (<=160 chars per rubric)
            if explanation_len <= 160:
                style_score += 0.15
                if explanation_len <= 120:
                    style_score += 0.05
            else:
                style_score -= min(0.001 * (explanation_len - 160), 0.2)
            # discourage multi-line prose
            if "\n" in explanation:
                style_score -= 0.05
        else:
            invalid_solution = 1.0
            structure_score -= 0.25
            traj.metadata["parse_error"] = "explanation not string"

        # Correctness only if we have a prediction
        if predicted in (0, 1):
            truth = 1 if example.label else 0
            correctness = 1.0 if predicted == truth else 0.0

    # Aggregate reward: correctness dominates, structure/style shape gradients.
    reward = (correctness * 1.2) + structure_score + style_score

    # Log scalar metadata
    traj.metadata["predicted"] = int(predicted)
    traj.metadata["explanation_len"] = int(explanation_len)
    traj.metadata["content_len"] = int(len(content))

    # Numeric metrics for training and analysis
    traj.metrics["correctness"] = float(correctness)
    traj.metrics["json_valid"] = float(json_valid)
    traj.metrics["invalid_solution"] = float(invalid_solution)
    traj.metrics["extra_keys"] = float(extra_keys)
    traj.metrics["structure_score"] = float(structure_score)
    traj.metrics["style_score"] = float(style_score)
    traj.metrics["reward"] = float(reward)

    traj.reward = float(reward)
    return traj

