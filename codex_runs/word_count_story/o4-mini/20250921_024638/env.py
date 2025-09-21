"""
Environment configuration and utilities for short-story word-count RL task.
"""
from __future__ import annotations
import random
from typing import TypedDict, List

# Random seed for reproducibility
RANDOM_SEED: int = 42

class StoryExample(TypedDict):
    """Reference example for a short story scenario."""
    theme: str
    word_count: int
    reference: str

# Seed with 10 reference prompts and exact-solution stories
STORY_EXAMPLES: List[StoryExample] = [
    {"theme": "friendship", "word_count": 10,
     "reference": "Two friends walked together under sunset skies, their bond unbroken."},
    {"theme": "mystery", "word_count": 12,
     "reference": "A locked door hid a secret world beneath, waiting for someone curious."},
    {"theme": "adventure", "word_count": 15,
     "reference": "Brave explorer journeyed through dense jungle, encountering wild creatures and hidden dangers at every turn."},
    {"theme": "hope", "word_count": 8,
     "reference": "Tiny seed sprouted through concrete cracks, reaching sunlight."},
    {"theme": "loss", "word_count": 9,
     "reference": "Old photo faded, edges framing memories of lost laughter."},
    {"theme": "joy", "word_count": 7,
     "reference": "Children danced barefoot in meadow, laughter echoing."},
    {"theme": "courage", "word_count": 11,
     "reference": "Heart pounding, she stepped onto the stage, voice trembling yet resolute."},
    {"theme": "betrayal", "word_count": 14,
     "reference": "Old friend’s smile cracked as secrets spilled, trust shattered into irreparable fragments of memory."},
    {"theme": "wonder", "word_count": 13,
     "reference": "Gazing at star field, she felt infinitesimal yet profoundly connected to cosmic dance."},
    {"theme": "homecoming", "word_count": 16,
     "reference": "Returning soldier’s footsteps echoed on familiar streets, each landmark greeting him with silent stories of yesterday."},
]

def get_example(step: int) -> StoryExample:
    """Select a reference scenario by step index."""
    return STORY_EXAMPLES[step % len(STORY_EXAMPLES)]

# Training configuration parameters (tweakable)
TRAINING_CONFIG: dict = {
    "project": "short-story-wordcount",
    "model_name": "story-agent",
    "base_model": "gpt-3.5-turbo",
    "steps": 100,
    "trajectories_per_group": 16,
    "groups_per_step": 1,
    "learning_rate": 1e-5,
    "max_completion_tokens": 64,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_exceptions": 4,
    "cleanup_keep_last": 1,
}
 
def init_seed() -> None:
    """Initialize random seed for reproducibility."""
    random.seed(RANDOM_SEED)
