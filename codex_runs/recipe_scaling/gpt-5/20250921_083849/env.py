"""Shared environment and helpers for the Recipe Scaling ART task.

This module centralizes tweakable constants, base recipes, and prompt helpers.
Keep hyperparameters near the top for easy modification.

Assumptions:
- The host project wires up training; we only expose config and rollout helpers.
- Inference/training uses a LocalBackend; no GPU selection logic here.

Installation (if needed):
- Use uv for dependencies: `uv pip install openai requests weave` and ART deps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, TypedDict


# -----------------------------
# Tunables & training constants
# -----------------------------
RANDOM_SEED: int = 7

# Minimal training config expected by host training harness.
TRAINING_CONFIG: dict[str, Any] = {
    "project": "recipe-scaling",
    "model_name": "agent-recipe-001",
    "base_model": "Qwen/Qwen2.5-1.5B",
    "steps": 30,
    "trajectories_per_group": 12,
    "groups_per_step": 1,
    "learning_rate": 2e-5,
    "max_completion_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_exceptions": 12,
    # Keep last N checkpoints when cleaning up between steps
    "cleanup_keep_last": 1,
}


# -----------------------------
# Recipe data model & registry
# -----------------------------
class Ingredient(TypedDict):
    name: str
    quantity: float
    unit: str


class Recipe(TypedDict):
    name: str
    servings: int
    ingredients: List[Ingredient]


# A small pool of base recipes. Quantities are realistic but simple.
# Units are canonicalized to a short form to ease validation.
BASE_RECIPES: list[Recipe] = [
    {
        "name": "Fluffy Pancakes",
        "servings": 4,
        "ingredients": [
            {"name": "all-purpose flour", "quantity": 200.0, "unit": "g"},
            {"name": "milk", "quantity": 300.0, "unit": "ml"},
            {"name": "egg", "quantity": 2.0, "unit": "pc"},
            {"name": "sugar", "quantity": 25.0, "unit": "g"},
            {"name": "butter (melted)", "quantity": 30.0, "unit": "g"},
            {"name": "salt", "quantity": 2.0, "unit": "g"},
            {"name": "baking powder", "quantity": 10.0, "unit": "g"},
        ],
    },
    {
        "name": "Tomato Soup",
        "servings": 3,
        "ingredients": [
            {"name": "tomato", "quantity": 600.0, "unit": "g"},
            {"name": "vegetable stock", "quantity": 500.0, "unit": "ml"},
            {"name": "onion", "quantity": 100.0, "unit": "g"},
            {"name": "olive oil", "quantity": 15.0, "unit": "ml"},
            {"name": "salt", "quantity": 4.0, "unit": "g"},
            {"name": "black pepper", "quantity": 1.0, "unit": "g"},
        ],
    },
    {
        "name": "Simple Pasta",
        "servings": 2,
        "ingredients": [
            {"name": "spaghetti", "quantity": 180.0, "unit": "g"},
            {"name": "garlic", "quantity": 6.0, "unit": "g"},
            {"name": "olive oil", "quantity": 20.0, "unit": "ml"},
            {"name": "parmesan", "quantity": 30.0, "unit": "g"},
            {"name": "salt", "quantity": 3.0, "unit": "g"},
            {"name": "black pepper", "quantity": 1.0, "unit": "g"},
        ],
    },
]


# -----------------------------
# Unit and name normalization
# -----------------------------
UNIT_SYNONYMS: dict[str, str] = {
    # mass
    "gram": "g",
    "grams": "g",
    "g": "g",
    # volume
    "milliliter": "ml",
    "milliliters": "ml",
    "ml": "ml",
    "tablespoon": "tbsp",
    "tablespoons": "tbsp",
    "tbsp": "tbsp",
    "teaspoon": "tsp",
    "teaspoons": "tsp",
    "tsp": "tsp",
    # count
    "pc": "pc",
    "piece": "pc",
    "pieces": "pc",
    "egg": "pc",
    "eggs": "pc",
}


def canonical_unit(unit: str) -> str:
    """Return a canonical short unit string (lowercase)."""
    u = unit.strip().lower()
    return UNIT_SYNONYMS.get(u, u)


def normalize_name(name: str) -> str:
    """Lowercase, strip, and collapse spaces & punctuation for stable matching."""
    s = name.lower().strip()
    # Keep letters/numbers/spaces, replace others with space, then collapse
    out = []
    prev_space = False
    for ch in s:
        if ch.isalnum() or ch.isspace():
            if ch.isspace():
                if not prev_space:
                    out.append(" ")
                prev_space = True
            else:
                out.append(ch)
                prev_space = False
        else:
            if not prev_space:
                out.append(" ")
                prev_space = True
    return "".join(out).strip()


# -----------------------------
# Prompt helpers
# -----------------------------
SYSTEM_PROMPT: str = (
    "You scale cooking recipes. Respond ONLY with JSON. "
    "JSON schema: {\"servings\": int, \"ingredients\": ["
    "{\"name\": str, \"quantity\": number, \"unit\": str}]}. "
    "No explanations, no code fences, no prose."
)


def build_user_prompt(recipe: Recipe, target_servings: int) -> str:
    """Create a compact user prompt describing the scaling task.

    The agent must scale all ingredient quantities proportionally from the
    provided base servings to the requested target servings and return strict
    JSON per SYSTEM_PROMPT.
    """
    lines = [
        f"Base recipe: {recipe['name']}",
        f"Original servings: {recipe['servings']}",
        "Ingredients (name, quantity, unit):",
    ]
    for ing in recipe["ingredients"]:
        lines.append(f"- {ing['name']} | {ing['quantity']} | {ing['unit']}")
    lines.append(f"Target servings: {target_servings}")
    lines.append(
        "Return JSON with keys 'servings' and 'ingredients' only; quantities as numbers."
    )
    return "\n".join(lines)


def recipe_for_step(step: int) -> Recipe:
    """Deterministically select a recipe from BASE_RECIPES for the given step."""
    idx = step % len(BASE_RECIPES)
    recipe = BASE_RECIPES[idx]
    # Canonicalize units for safety
    canonicalized: Recipe = {
        "name": recipe["name"],
        "servings": int(recipe["servings"]),
        "ingredients": [
            {
                "name": i["name"],
                "quantity": float(i["quantity"]),
                "unit": canonical_unit(i["unit"]),
            }
            for i in recipe["ingredients"]
        ],
    }
    return canonicalized


def suggested_target_servings(original: int, step: int) -> int:
    """Pick a simple but varying target servings value.

    Uses the step to avoid randomness: cycles through a small, interesting set.
    """
    # Values chosen to hit <1x, ~1x, and >1x scaling ratios.
    options = [1, 2, 3, 4, 6, 8, 10, 12]
    # Avoid asking for the same value; pick the next option.
    try:
        j = (options.index(original) + (1 + (step % 3))) % len(options)
    except ValueError:
        j = (step + original) % len(options)
    target = options[j]
    return target


__all__ = [
    "RANDOM_SEED",
    "TRAINING_CONFIG",
    "Ingredient",
    "Recipe",
    "BASE_RECIPES",
    "SYSTEM_PROMPT",
    "build_user_prompt",
    "recipe_for_step",
    "normalize_name",
    "canonical_unit",
    "suggested_target_servings",
]

