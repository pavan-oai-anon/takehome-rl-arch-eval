"""Rollout implementation for ART reinforcement learning task."""
from typing import Any, Dict
import random
import asyncio
import art
import weave
from env import TRAINING_CONFIG, get_prompt

@weave.op
@art.retry(exceptions=(art.exceptions.StepTimeoutError,))
async def rollout(model: art.Model, step: int, config: Dict[str, Any]) -> art.Trajectory:
    """Perform a rollout using the provided model and environment setup."""
    prompt_data = get_prompt(step % len(get_prompt))  # Circular access to prompts
    prompt = (f"Write a {prompt_data['word_count']} word story about a(n) {prompt_data['theme']}. ")

    trajectory = art.Trajectory(
        messages_and_choices=[
            {"role": "system", "content": "You are a skilled storyteller."},
            {"role": "user", "content": prompt}
        ],
        metadata={"step": step},
        reward=0,
    )

    client = art.AsyncModel(model=model)
    try:
        response = await client.generate(prompt, max_tokens=config['max_completion_tokens'])
        story = response.content.strip()
        word_count = len(story.split())

        trajectory.messages_and_choices.append({"role": "assistant", "content": story})
        trajectory.metrics = {"word_count_error": abs(word_count - prompt_data['word_count'])}
        if word_count == prompt_data['word_count']:
            trajectory.reward = 1  # perfect match
        else:
            trajectory.reward = 1 - (trajectory.metrics['word_count_error'] / prompt_data['word_count'])

    except Exception as e:
        trajectory.reward = -1
        trajectory.metadata['error'] = str(e)

    return trajectory
