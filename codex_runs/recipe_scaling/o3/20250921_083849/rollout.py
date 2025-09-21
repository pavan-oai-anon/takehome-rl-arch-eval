"""ART rollout logic for the *recipe-scaling* reinforcement-learning task.

The agent receives a recipe and a *target* number of servings. It must respond
with **only** a JSON payload – a list of ingredient dicts each holding
`name`, `quantity`, and `unit`. We then parse / validate that payload, measure
the per-ingredient scaling error, and shape a reward:

    + 1.0  – perfect scaling (absolute relative error < 1 × 10⁻⁶).
    + 0.5  – average relative error ≤ 5 % and no missing / extra ingredients.
    – 1.0  – anything else (parse errors, wrong units, missing ingredients …).

All numeric debug signals are surfaced through *scalar* metadata / metrics so
that ART can aggregate them across workers.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, Tuple

import art
import requests
import weave
from openai import AsyncOpenAI

import env  # local module holding constants & helpers


_JSON_RE = re.compile(r"```(?:json)?\s*(?P<json>{[\s\S]*?}|\[[\s\S]*?])\s*```", re.IGNORECASE)


def _extract_json(text: str) -> str | None:
    """Return the first JSON blob found inside *text* (or None)."""

    match = _JSON_RE.search(text)
    if match:
        return match.group("json")

    # Fallback – maybe the assistant sent raw JSON with no fencing.
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        return text
    if text.startswith("{") and text.endswith("}"):
        return text
    return None


def _relative_error(true_val: float, pred_val: float) -> float:
    """Return |true - pred| / true, guarding against division-by-zero."""

    if true_val == 0:
        return 0.0 if pred_val == 0 else 1.0
    return abs(true_val - pred_val) / true_val


def _evaluate_prediction(
    recipe: env.Recipe, prediction: list[Dict[str, Any]]
) -> Tuple[float, Dict[str, float]]:
    """Compare *prediction* to ground-truth scaled recipe.

    Returns (reward, scalar_metrics).
    """

    # Map by lowercase name for robust matching.
    gt_map = {
        ing.name.lower(): ing
        for ing in recipe.ingredients
    }
    pred_map = {
        (ing["name"] if isinstance(ing, dict) else "").lower(): ing
        for ing in prediction
        if isinstance(ing, dict)
    }

    missing = [name for name in gt_map if name not in pred_map]
    extra = [name for name in pred_map if name not in gt_map]

    # If any ingredient is missing or extra – penalise heavy.
    if missing or extra:
        return -1.0, {
            "avg_pct_error": 1.0,
            "missing_flag": 1.0,
            "extra_flag": 1.0,
        }

    # Compute average relative error across quantities.
    errors = []
    unit_mismatch = False
    for name, gt in gt_map.items():
        pred = pred_map[name]

        try:
            pred_qty = float(pred["quantity"])
            pred_unit = str(pred["unit"]).strip().lower()
        except (KeyError, TypeError, ValueError):
            return -1.0, {
                "avg_pct_error": 1.0,
                "missing_flag": 0.0,
                "extra_flag": 0.0,
            }

        if pred_unit != gt.unit.lower():
            unit_mismatch = True
        errors.append(_relative_error(gt.quantity, pred_qty))

    avg_error = float(sum(errors) / len(errors)) if errors else 1.0

    if unit_mismatch:
        return -1.0, {
            "avg_pct_error": avg_error,
            "missing_flag": 0.0,
            "extra_flag": 0.0,
        }

    if avg_error < 1e-6:
        reward = 1.0
    elif avg_error <= 0.05:
        reward = 0.5
    else:
        reward = -1.0

    return reward, {
        "avg_pct_error": avg_error,
        "missing_flag": 0.0,
        "extra_flag": 0.0,
    }


# ART exposes the decorator at the top-level.
# noqa: decorator-position – want retry *inside* weave wrapper like example.
@weave.op
@art.retry(exceptions=(requests.ReadTimeout,))
async def rollout(model: art.Model, step: int, config: Dict[str, Any]) -> art.Trajectory:  # type: ignore[override]
    """Single training episode – query the agent and evaluate its scaling logic."""

    rng = random.Random(env.RANDOM_SEED + step)
    recipe = config.get("recipe") or env.random_recipe(rng=rng)
    target_servings = config.get("target_servings") or rng.choice(list(env.valid_serving_sizes(recipe)))

    ground_truth = recipe.scale_to(target_servings)

    # Construct conversation
    system_msg = (
        "You are an expert chef. When asked, scale *all* ingredient quantities "
        "proportionally to match the requested servings. Respond with **only** "
        "JSON – a list of objects holding name, quantity (number), and unit. "
        "Do *not* wrap the JSON in any extra text."
    )

    user_msg = (
        f"Here is a recipe for {recipe.name} that yields {recipe.servings} servings:\n"  # noqa: E501
        + "\n".join(
            f"- {ing.name}: {ing.quantity} {ing.unit}" for ing in recipe.ingredients
        )
        + f"\n\nPlease scale it to produce exactly {target_servings} servings."
    )

    trajectory = art.Trajectory(
        messages_and_choices=[{"role": "system", "content": system_msg}],
        metadata={
            "recipe_name": recipe.name,
            "orig_servings": recipe.servings,
            "target_servings": target_servings,
            "step": step,
        },
        reward=0.0,
    )

    trajectory.messages_and_choices.append({"role": "user", "content": user_msg})

    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )

    chat_completion = await client.chat.completions.create(
        model=model.name,
        max_completion_tokens=env.TRAINING_CONFIG["max_completion_tokens"],
        temperature=env.TRAINING_CONFIG["temperature"],
        top_p=env.TRAINING_CONFIG["top_p"],
        stream=False,
        messages=trajectory.messages(),
    )

    choice = chat_completion.choices[0]
    content = choice.message.content or ""
    trajectory.messages_and_choices.append(choice)

    # ---------------------------------------------------------------------
    # Parse / validate assistant output
    # ---------------------------------------------------------------------
    parse_error = 0.0
    try:
        json_blob = _extract_json(content)
        if json_blob is None:
            raise ValueError("No JSON found in assistant reply")
        parsed = json.loads(json_blob)
        if not isinstance(parsed, list):
            raise ValueError("Top-level JSON must be a list of ingredients")
    except Exception:  # noqa: BLE001 – defensive catch-all
        trajectory.reward = -1.0
        parse_error = 1.0
        trajectory.metadata["parse_error"] = 1.0
        trajectory.metrics["avg_pct_error"] = 1.0
        return trajectory

    reward, metrics = _evaluate_prediction(ground_truth, parsed)  # type: ignore[arg-type]

    trajectory.reward = reward
    trajectory.metrics.update(metrics)
    trajectory.metadata.update({k: v for k, v in metrics.items()})
    trajectory.metadata["parse_error"] = parse_error

    return trajectory
