"""Rollout logic for Tic-Tac-Toe ART RL setup."""
from __future__ import annotations

import random
import art
import weave
from openai import AsyncOpenAI
from typing import Any

from env import (
    create_board,
    render_board,
    is_valid_move,
    apply_move,
    check_winner,
    PLAYER_X,
    PLAYER_O,
    board_hash,
)

@weave.op
@art.retry(exceptions=(Exception,))  # retry on generic errors
async def rollout(model: art.Model, step: int, config: dict[str, Any]) -> art.Trajectory:
    """Generate one Tic-Tac-Toe episode rollout with random opponent."""
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )

    board = create_board()
    # convert hash to numeric scalar
    bh = int(board_hash(board), 16) % (10 ** 8)
    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": (
                    "You are a Tic-Tac-Toe agent. Board rows A-C, cols 1-3, '.' empty.\n"
                    "Respond with a single move like 'B2'."
                ),
            }
        ],
        metadata={
            "step": float(step),
            "board_hash": float(bh),
            "illegal": 0.0,
            "winner": 0.0,
        },
        reward=0.0,
    )
    while True:
        # Agent's turn
        trajectory.messages_and_choices.append(
            {"role": "user", "content": render_board(board)}
        )
        response = await client.chat.completions.create(
            max_completion_tokens=config["max_completion_tokens"],
            messages=trajectory.messages(),
            model=model.name,
            temperature=config["temperature"],
            top_p=config["top_p"],
            stream=False,
        )
        choice = response.choices[0]
        content = choice.message.content.strip()
        trajectory.messages_and_choices.append(choice)
        move = content.upper()
        # validate move
        if not is_valid_move(board, move):
            trajectory.metadata["illegal"] = 1.0
            trajectory.metrics["invalid_solution"] = 1.0
            trajectory.reward = -1.0
            break
        apply_move(board, move, PLAYER_X)
        # check terminal
        result = check_winner(board)
        if result is not None:
            if result == PLAYER_X:
                trajectory.metadata["winner"] = 1.0
                trajectory.reward = 1.0
            elif result == PLAYER_O:
                trajectory.metadata["winner"] = -1.0
                trajectory.reward = -1.0
            else:
                trajectory.metadata["winner"] = 0.0
                trajectory.reward = 0.1
            break
        # Opponent plays randomly
        valid = [f"{r}{c}" for r in ["A","B","C"] for c in ["1","2","3"] if is_valid_move(board, f"{r}{c}")]
        if not valid:
            trajectory.metadata["winner"] = 0.0
            trajectory.reward = 0.1
            break
        opp = random.choice(valid)
        apply_move(board, opp, PLAYER_O)
        result = check_winner(board)
        if result is not None:
            if result == PLAYER_X:
                trajectory.metadata["winner"] = 1.0
                trajectory.reward = 1.0
            elif result == PLAYER_O:
                trajectory.metadata["winner"] = -1.0
                trajectory.reward = -1.0
            else:
                trajectory.metadata["winner"] = 0.0
                trajectory.reward = 0.1
            break
        # continue loop for next agent move
    return trajectory
