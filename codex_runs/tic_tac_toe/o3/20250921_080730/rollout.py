"""ART rollout logic for the Tic-Tac-Toe task.

This file is intentionally minimal – the heavy lifting happens in :pymod:`env`.
We only orchestrate the agent/environment interaction, perform defensive
validation, and convert the episode into an :class:`art.Trajectory` object that
the ART framework can consume for both training and evaluation.
"""

from __future__ import annotations

import random
import re
from typing import Any

import art
import weave
from openai import AsyncOpenAI

import env as ttt


# ---------------------------------------------------------------------------
# Rollout implementation
# ---------------------------------------------------------------------------


def _system_prompt() -> str:
    """Return the fixed system prompt for the policy."""

    return (
        "You are an expert Tic-Tac-Toe player. You play as X and move first. "
        "Respond *only* with the coordinate of your chosen move using row-column "
        "notation like B2 (rows A-C, columns 1-3). Do not add any explanation."
    )


_MOVE_RE = re.compile(r"^[ABCabc][123]$")


def _validate_move(raw: str) -> bool:
    """Return ``True`` if *raw* looks like a legal coordinate string."""

    return bool(_MOVE_RE.fullmatch(raw.strip()))


# We deliberately expose *step* and *config* even though the rollout does not
# require them – the function signature must remain stable for the generic
# training pipeline.


@weave.op
@art.retry(exceptions=(Exception,))
async def rollout(
    model: art.Model, step: int, config: dict[str, Any] | None = None
) -> art.Trajectory:  # noqa: D401 – simple signature expected by trainer
    """Play a full game where the agent is *X* against a random *O* opponent."""

    # Prepare inference client (LocalBackend expected during training).
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )

    board = ttt.empty_board()
    move_count = 0
    invalid_flag = 0.0  # numeric metric

    trajectory = art.Trajectory(
        messages_and_choices=[{"role": "system", "content": _system_prompt()}],
        metadata={
            "game_id": random.randint(0, 1_000_000),  # scalar, not list/dict
            "step": step,
        },
        reward=0.0,
    )

    # Helper to append board state as user message.
    def _push_board_message() -> None:
        trajectory.messages_and_choices.append({
            "role": "user",
            "content": ttt.render_board(board),
        })

    # Game loop – agent move then random opponent until terminal state.
    while True:
        _push_board_message()

        # Ask the model for its move.
        completion = await client.chat.completions.create(
            model=model.name,
            messages=trajectory.messages(),
            max_completion_tokens=ttt.TRAINING_CONFIG["max_completion_tokens"],
            temperature=ttt.TRAINING_CONFIG["temperature"],
            top_p=ttt.TRAINING_CONFIG["top_p"],
            stream=False,
        )

        choice = completion.choices[0]
        agent_move_raw = choice.message.content or ""
        trajectory.messages_and_choices.append(choice)

        # ------------------------------------------------------------------
        # Validation & application of the agent's move.
        # ------------------------------------------------------------------

        legal_move = _validate_move(agent_move_raw)
        if not legal_move:
            trajectory.reward = -1.0
            invalid_flag = 1.0
            break

        indices = ttt.parse_move(agent_move_raw)  # type: ignore[arg-type]
        assert indices is not None  # safe due to regex pass
        r, c = indices
        if board[r][c] != " ":
            # Occupied square → illegal.
            trajectory.reward = -1.0
            invalid_flag = 1.0
            break

        board[r][c] = "X"
        move_count += 1

        winner = ttt.check_winner(board)
        if winner or ttt.board_full(board):
            # Terminal state after agent move.
            outcome_reward = _outcome_reward(winner)
            trajectory.reward = outcome_reward
            break

        # Opponent (random) move.
        opp_r, opp_c = ttt.random_legal_move(board)
        board[opp_r][opp_c] = "O"
        move_count += 1

        winner = ttt.check_winner(board)
        if winner or ttt.board_full(board):
            outcome_reward = _outcome_reward(winner)
            trajectory.reward = outcome_reward
            break

    # ------------------------------------------------------------------
    # Metrics / metadata – must be scalar values only.
    # ------------------------------------------------------------------

    trajectory.metadata["board_hash"] = ttt.board_hash(board)
    trajectory.metadata["winner"] = winner or "draw"  # type: ignore[assignment]
    trajectory.metrics["invalid_move"] = invalid_flag
    trajectory.metrics["move_count"] = float(move_count)

    return trajectory


def _outcome_reward(winner: str | None) -> float:
    """Reward shaping based on the final outcome."""

    if winner == "X":
        return 1.0
    if winner == "O":
        return -1.0
    # Draw
    return 0.1

