"""Environment helpers and shared configuration for the Recipe-Scaling RL task.

This module is intentionally lightweight – it only exposes a few constants and
pure-Python helpers so that OpenPipe's ART framework can import it in any
context (GPU / CPU) without heavy dependencies. All hyper-parameters live at
the top for quick experimentation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List

# ---------------------------------------------------------------------------
# Public constants expected by ART tooling
# ---------------------------------------------------------------------------

# Fix the global RNG so that dataset generation is reproducible and
# deterministic across workers.
RANDOM_SEED: int = 2025
random.seed(RANDOM_SEED)

# Training hyper-parameters – tweak freely. The host project wires these into
# the generic `training.py` loop.
TRAINING_CONFIG: dict = {
    "project": "recipe-scaling",
    "model_name": "chef-agent-001",
    "base_model": "Qwen/Qwen2.5-1.5B",
    "steps": 30,
    "trajectories_per_group": 16,
    "groups_per_step": 1,
    "learning_rate": 5e-6,
    # Inference parameters – keep completions short & cheap.
    "max_completion_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    # Runtime robustness / housekeeping.
    "max_exceptions": 16,
    "cleanup_keep_last": 3,
}

# ---------------------------------------------------------------------------
# Domain helpers – basic recipe representation & math
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ingredient:
    """Single ingredient entry inside a recipe."""

    name: str
    quantity: float  # stored in *base units* (e.g. grams, millilitres)
    unit: str

    def scale(self, factor: float) -> "Ingredient":
        """Return a *new* Ingredient scaled by the provided factor."""

        return Ingredient(self.name, self.quantity * factor, self.unit)


@dataclass(frozen=True)
class Recipe:
    """Simple container describing a cooking recipe."""

    name: str
    servings: int
    ingredients: List[Ingredient]

    def scale_to(self, new_servings: int) -> "Recipe":
        """Return a *new* recipe scaled to ``new_servings``."""

        factor = new_servings / self.servings
        return Recipe(
            name=self.name,
            servings=new_servings,
            ingredients=[ing.scale(factor) for ing in self.ingredients],
        )


# A handful of tiny base recipes to keep the state-space manageable.
BASE_RECIPES: list[Recipe] = [
    Recipe(
        name="Pancakes",
        servings=4,
        ingredients=[
            Ingredient("all-purpose flour", 240, "g"),
            Ingredient("milk", 480, "ml"),
            Ingredient("egg", 2, "pc"),
            Ingredient("baking powder", 10, "g"),
            Ingredient("salt", 3, "g"),
            Ingredient("unsalted butter", 30, "g"),
        ],
    ),
    Recipe(
        name="Tomato Soup",
        servings=2,
        ingredients=[
            Ingredient("tomato", 400, "g"),
            Ingredient("onion", 100, "g"),
            Ingredient("vegetable stock", 500, "ml"),
            Ingredient("olive oil", 15, "ml"),
            Ingredient("salt", 4, "g"),
            Ingredient("black pepper", 1, "g"),
        ],
    ),
    Recipe(
        name="Guacamole",
        servings=3,
        ingredients=[
            Ingredient("avocado", 3, "pc"),
            Ingredient("lime juice", 30, "ml"),
            Ingredient("red onion", 40, "g"),
            Ingredient("cilantro", 5, "g"),
            Ingredient("salt", 3, "g"),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Convenience functions used by rollout.py
# ---------------------------------------------------------------------------


def random_recipe(*, rng: random.Random | None = None) -> Recipe:
    """Pick a random recipe from ``BASE_RECIPES`` using the provided RNG."""

    _rng = rng or random
    return _rng.choice(BASE_RECIPES)


def valid_serving_sizes(recipe: Recipe) -> Iterable[int]:
    """Yield plausible alternative serving sizes for the given recipe."""

    multiples = (0.5, 0.75, 2, 3)
    for multiplier in multiples:
        val = int(round(recipe.servings * multiplier))
        if val != recipe.servings and val > 0:
            yield val

