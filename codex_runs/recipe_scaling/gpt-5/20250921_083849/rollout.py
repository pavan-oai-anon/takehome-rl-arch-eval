"""ART rollout for the Recipe Scaling task.

This rollout asks the model to scale a base recipe to a target serving count,
returning strict JSON. We compute per-ingredient percent errors, apply
defensive validation, and shape rewards to be informative while honoring the
requested rules:
- Exact scaling => reward +1.0
- If max abs % error < 5% and formatting valid => +0.5
- Missing ingredients or invalid formatting => -1.0

We also record scalar metadata for diagnostics and numeric metrics for ART.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, Tuple

import art
import requests
import weave
from openai import AsyncOpenAI

from env import (
    RANDOM_SEED,
    TRAINING_CONFIG,
    SYSTEM_PROMPT,
    build_user_prompt,
    recipe_for_step,
    normalize_name,
    canonical_unit,
    suggested_target_servings,
)


def _safe_float(x: Any) -> Tuple[float, bool]:
    """Try to interpret x as float. Return (value, ok)."""
    try:
        if isinstance(x, bool):  # exclude booleans
            return 0.0, False
        return float(x), True
    except Exception:
        return 0.0, False


def _parse_agent_json(text: str) -> Tuple[dict[str, Any] | None, str | None]:
    """Extract a JSON object from the assistant response.

    Returns (data, error_message). If parsing fails or schema invalid, data is None.
    """
    s = text.strip()
    # Heuristic: if model included code fences, strip them
    if s.startswith("```") and s.endswith("```"):
        s = s.strip("`")
        # remove optional language tag
        s = s.split('\n', 1)[-1]

    try:
        data = json.loads(s)
    except Exception as exc:
        return None, f"json_parse_error: {type(exc).__name__}"

    if not isinstance(data, dict):
        return None, "top_level_not_object"
    if "ingredients" not in data or "servings" not in data:
        return None, "missing_required_keys"
    ings = data.get("ingredients")
    if not isinstance(ings, list) or len(ings) == 0:
        return None, "ingredients_not_list_or_empty"

    # Basic ingredient validation; we allow extra fields but require core keys.
    for idx, ing in enumerate(ings):
        if not isinstance(ing, dict):
            return None, f"ingredient_{idx}_not_object"
        if not all(k in ing for k in ("name", "quantity", "unit")):
            return None, f"ingredient_{idx}_missing_keys"
        qv, ok = _safe_float(ing["quantity"])
        if not ok:
            return None, f"ingredient_{idx}_quantity_not_number"
        if not isinstance(ing["name"], str) or not isinstance(ing["unit"], str):
            return None, f"ingredient_{idx}_name_or_unit_not_string"

    return data, None


def _percent_error(expected: float, actual: float) -> float:
    denom = max(abs(expected), 1e-9)
    return abs(actual - expected) / denom * 100.0


@weave.op
@art.retry(exceptions=(requests.ReadTimeout,))
async def rollout(model: art.Model, step: int, config: dict[str, Any]) -> art.Trajectory:
    """Collect a single scaling attempt trajectory for a selected recipe.

    The config argument is merged with TRAINING_CONFIG for decoding options.
    """
    # Merge decode config with defaults
    cfg = {**TRAINING_CONFIG, **(config or {})}

    # Prepare scenario
    recipe = recipe_for_step(step)
    target_servings = suggested_target_servings(recipe["servings"], step)
    factor = float(target_servings) / float(recipe["servings"])

    # Build messages
    trajectory = art.Trajectory(
        messages_and_choices=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(recipe, target_servings)},
        ],
        metadata={
            # Scalar-only metadata for ART aggregation
            "task": "recipe-scaling",
            "recipe_name": recipe["name"],
            "step": step,
            "orig_servings": recipe["servings"],
            "target_servings": target_servings,
            "scale_factor": factor,
            "invalid_solution": 0.0,
            "missing_any": 0.0,
            "extra_any": 0.0,
            "parse_error_flag": 0.0,
        },
        reward=0.0,
    )

    client = AsyncOpenAI(base_url=model.inference_base_url, api_key=model.inference_api_key)

    # Model inference
    completion = await client.chat.completions.create(
        model=model.name,
        messages=trajectory.messages(),
        max_completion_tokens=int(cfg.get("max_completion_tokens", 256)),
        temperature=float(cfg.get("temperature", 0.7)),
        top_p=float(cfg.get("top_p", 0.9)),
        stream=False,
    )

    choice = completion.choices[0]
    content = choice.message.content or ""
    trajectory.messages_and_choices.append(choice)

    # Parse & validate
    data, parse_err = _parse_agent_json(content)
    if parse_err:
        trajectory.metadata["parse_error"] = parse_err
        trajectory.metadata["parse_error_flag"] = 1.0
        trajectory.metadata["invalid_solution"] = 1.0
        trajectory.reward = -1.0
        # Minimal metrics for failed parse
        trajectory.metrics["mean_abs_pct_error"] = 100.0
        trajectory.metrics["max_abs_pct_error"] = 100.0
        trajectory.metrics["matched_count"] = 0.0
        trajectory.metrics["omitted_count"] = float(len(recipe["ingredients"]))
        trajectory.metrics["extra_count"] = 0.0
        return trajectory

    # Build indices for comparison
    base_map: dict[str, tuple[float, str]] = {}
    for ing in recipe["ingredients"]:
        base_map[normalize_name(ing["name"])[:48]] = (float(ing["quantity"]), canonical_unit(ing["unit"]))

    agent_ings = data.get("ingredients", [])
    agent_map: dict[str, tuple[float, str, str]] = {}  # norm_name -> (qty, unit, raw_name)
    for ing in agent_ings:
        nm = normalize_name(str(ing["name"]))[:48]
        qty, _ = _safe_float(ing["quantity"])  # already validated
        agent_map[nm] = (qty, canonical_unit(str(ing["unit"])), str(ing["name"]))

    # Compare
    omitted = 0
    extra = 0
    unit_mismatch = 0
    errors: list[float] = []
    max_err = 0.0
    matched = 0

    for norm_name, (base_qty, base_unit) in base_map.items():
        expected = base_qty * factor
        if norm_name not in agent_map:
            omitted += 1
            trajectory.metadata[f"err_{norm_name}_pct"] = 100.0  # missing
            continue
        qty, unit, _raw = agent_map[norm_name]
        if unit != base_unit:
            unit_mismatch += 1
        e = _percent_error(expected, qty)
        errors.append(e)
        matched += 1
        max_err = max(max_err, e)
        trajectory.metadata[f"err_{norm_name}_pct"] = float(e)

    for norm_name in agent_map.keys():
        if norm_name not in base_map:
            extra += 1

    # Metrics
    mean_err = float(sum(errors) / len(errors)) if errors else (100.0 if omitted else 0.0)
    trajectory.metrics["mean_abs_pct_error"] = mean_err
    trajectory.metrics["max_abs_pct_error"] = float(max_err if errors else (100.0 if omitted else 0.0))
    trajectory.metrics["matched_count"] = float(matched)
    trajectory.metrics["omitted_count"] = float(omitted)
    trajectory.metrics["extra_count"] = float(extra)
    trajectory.metrics["unit_mismatch_count"] = float(unit_mismatch)

    # Metadata flags (scalar-only)
    trajectory.metadata["missing_any"] = 1.0 if omitted > 0 else 0.0
    trajectory.metadata["extra_any"] = 1.0 if extra > 0 else 0.0
    trajectory.metadata["unit_mismatch_any"] = 1.0 if unit_mismatch > 0 else 0.0

    # Reward logic per spec with gentle shaping. Never exceed [−1, 1].
    if omitted > 0:
        trajectory.metadata["invalid_solution"] = 1.0
        reward = -1.0  # Missing required ingredients is a hard failure
    else:
        reward = 0.0
        if max_err < 0.5:
            reward = 1.0
        elif max_err < 5.0:
            reward = 0.5

        # Smooth shaping: small bonus for low mean error, small penalty for extras.
        # This keeps gradients informative without violating the primary rules.
        shaping = max(0.0, 0.3 * (1.0 - min(mean_err / 10.0, 1.0)))
        reward = min(1.0, reward + shaping)
        reward -= min(0.3, 0.05 * extra)

        # If schema is off but parsed, penalize strongly (should be caught above).
        # Kept for clarity and future extension.
    trajectory.reward = float(max(-1.0, min(1.0, reward)))

    # Record the reported servings if present
    srv_val, srv_ok = _safe_float(data.get("servings"))
    if srv_ok:
        trajectory.metrics["agent_servings"] = srv_val
        trajectory.metadata["agent_servings_int"] = int(round(srv_val))
    else:
        trajectory.metadata["agent_servings_parse_fail"] = 1.0

    return trajectory

