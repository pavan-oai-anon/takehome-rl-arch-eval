"""Rollout routines for collecting trajectories in the 24 game."""
from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET

import art
import requests
import weave
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict

from .config import TrainingConfig
from .env import (
    generate_game,
    render_puzzle,
    score_expression,
    validate_solution_expression,
    InvalidExpressionError,
)

SYSTEM_MESSAGE = (
    "You solve the 24 game. Given four digits, combine each digit exactly once "
    "using +, -, *, / to reach 24. Respond with XML like "
    "<solution>(3 * (4 + 4))</solution>. Only emit the XML wrapper."
)


class Scenario24(BaseModel):
    """Scenario data passed to the rollout operator."""

    step: int
    config: TrainingConfig

    model_config = ConfigDict(arbitrary_types_allowed=True)


def extract_expression(payload: str) -> str:
    """Pull the arithmetic expression out of the XML payload."""

    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:  # pragma: no cover - defensive
        raise InvalidExpressionError("Response must be valid XML with <solution> tag") from exc

    if root.tag != "solution":
        raise InvalidExpressionError("Expected <solution>...</solution> in the response")

    if root.text is None:
        raise InvalidExpressionError("Solution tag must contain an arithmetic expression")

    expression = root.text.strip()
    if not expression:
        raise InvalidExpressionError("Empty expression provided")

    return expression


@weave.op
@art.retry(exceptions=(requests.ReadTimeout, asyncio.TimeoutError))
async def rollout(model: art.Model, scenario: Scenario24) -> art.Trajectory:
    config = scenario.config
    client = AsyncOpenAI(base_url=model.inference_base_url, api_key=model.inference_api_key)

    game = generate_game()
    trajectory = art.Trajectory(
        messages_and_choices=[{"role": "system", "content": SYSTEM_MESSAGE}],
        metadata={
            "puzzle_id": game.id,
            "digits": game.digits,
            "step": scenario.step,
        },
        reward=0.0,
    )

    puzzle_prompt = render_puzzle(game)
    trajectory.messages_and_choices.append({"role": "user", "content": puzzle_prompt})

    chat_completion = await client.chat.completions.create(
        messages=trajectory.messages(),
        model=model.name,
        max_completion_tokens=config.max_completion_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        stream=False,
    )

    choice = chat_completion.choices[0]
    content = choice.message.content or ""
    trajectory.messages_and_choices.append(choice)

    try:
        expression = extract_expression(content)
        value = validate_solution_expression(expression, game.digits)
        metrics = score_expression(value, game.target)
        trajectory.metrics.update(metrics)
        if metrics["difference"] == 0:
            trajectory.reward = 2.0
        else:
            trajectory.reward = max(0.1, metrics["normalized_reward"])  # encourage near misses
    except InvalidExpressionError as error:
        trajectory.reward = -1.0
        trajectory.metrics["error"] = str(error)

    return trajectory
