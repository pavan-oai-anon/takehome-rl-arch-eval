"""ART rollout for Tic-Tac-Toe built on env.py.

This file defines a single rollout() op. It collects one full self-contained
game where the agent (X) plays against a built-in opponent (O). Rewards:
- +1 for win, -1 for loss, +0.1 for draw, -1 for illegal move.
- Small shaping is added to encourage center control, creating threats,
  and blocking opponent threats.

We assume a LocalBackend for inference/training. If you copy memory tuning
values from 2048.py, ensure they match your GPU and base model.
"""
from __future__ import annotations

import math
import random
from typing import Any

import art
import weave
from openai import AsyncOpenAI

from env import (
    RANDOM_SEED,
    TRAINING_CONFIG,
    AGENT_MARK,
    OPP_MARK,
    new_game,
    board_to_text,
    is_legal_move,
    parse_move,
    apply_move,
    outcome,
    board_hash,
    opponent_policy,
    shaping_after_agent_move,
    TTTState,
)


SYSTEM_PROMPT = (
    "You are a precise Tic-Tac-Toe player. Respond with exactly one move "
    "in row/column notation like B2. Use a single uppercase letter A-C and "
    "a single digit 1-3. Do not add punctuation or explanation."
)


def _extract_conf(config: dict[str, Any]) -> dict[str, Any]:
    """Merge runtime config with TRAINING_CONFIG defaults."""
    base = dict(TRAINING_CONFIG)
    base.update(config or {})
    return base


@weave.op
@art.retry(exceptions=(Exception,))
async def rollout(model: art.Model, step: int, config: dict[str, Any]) -> art.Trajectory:
    conf = _extract_conf(config)
    rnd = random.Random(RANDOM_SEED + int(step))

    client = AsyncOpenAI(base_url=model.inference_base_url, api_key=model.inference_api_key)

    # Initialize game state
    state: TTTState = new_game()
    agent_turns = 0
    shaping_sum = 0.0
    invalid_solution = 0.0
    illegal_flag = 0
    final_outcome = ""

    traj = art.Trajectory(
        messages_and_choices=[{"role": "system", "content": SYSTEM_PROMPT}],
        metadata={
            "game_id": state.id,
            "board_hash": board_hash(state),
            "winner": "",
            "illegal": 0,
            "step": int(step),
        },
        reward=0.0,
    )

    # Play until terminal or illegal
    while True:
        # Build a compact, repeatable user prompt.
        user_msg = (
            f"Board:\n{board_to_text(state)}\n"
            f"Turn: {state.turn}\nLast: {state.last_move or 'None'}\n"
            "Reply with a single coordinate like B2."
        )
        traj.messages_and_choices.append({"role": "user", "content": user_msg})

        # Query the model
        chat = await client.chat.completions.create(
            model=model.name,
            messages=traj.messages(),
            max_completion_tokens=int(conf.get("max_completion_tokens", 8)),
            temperature=float(conf.get("temperature", 0.7)),
            top_p=float(conf.get("top_p", 0.9)),
            stream=False,
        )
        choice = chat.choices[0]
        content = choice.message.content or ""
        traj.messages_and_choices.append(choice)

        # Validate and apply agent move
        try:
            move_str = content.strip().upper()
            if not is_legal_move(state, move_str):
                raise ValueError("illegal or malformed move")

            # Snapshot for shaping
            before = TTTState(board=[row[:] for row in state.board], turn=state.turn, last_move=state.last_move, id=state.id)
            apply_move(state, move_str, AGENT_MARK)
            after = state
            agent_turns += 1
            shaping_sum += shaping_after_agent_move(before, after, move_str)
        except Exception:
            invalid_solution = 1.0
            illegal_flag = 1
            final_outcome = "illegal"
            traj.metadata["illegal"] = 1
            traj.metadata["winner"] = final_outcome
            traj.metadata["board_hash"] = board_hash(state)
            traj.metrics["invalid_solution"] = invalid_solution
            traj.metrics["illegal_move"] = 1
            traj.metrics["turns"] = agent_turns
            traj.metrics["reward_outcome"] = -1.0
            traj.metrics["reward_shaping"] = 0.0
            traj.reward = -1.0
            return traj

        # Check terminal after agent move
        term = outcome(state)
        if term:
            final_outcome = term
            base = 1.0 if term == AGENT_MARK else (0.1 if term == "draw" else -1.0)
            total = max(-1.0, min(1.0, base + shaping_sum))
            traj.metadata["winner"] = term
            traj.metadata["board_hash"] = board_hash(state)
            traj.metrics["invalid_solution"] = invalid_solution
            traj.metrics["illegal_move"] = illegal_flag
            traj.metrics["turns"] = agent_turns
            traj.metrics["reward_outcome"] = base
            traj.metrics["reward_shaping"] = shaping_sum
            traj.reward = total
            return traj

        # Opponent move
        opp_mv = opponent_policy(state, rnd)
        apply_move(state, opp_mv, OPP_MARK)

        # Check terminal after opponent move
        term = outcome(state)
        if term:
            final_outcome = term
            base = 1.0 if term == AGENT_MARK else (0.1 if term == "draw" else -1.0)
            total = max(-1.0, min(1.0, base + shaping_sum))
            traj.metadata["winner"] = term
            traj.metadata["board_hash"] = board_hash(state)
            traj.metrics["invalid_solution"] = invalid_solution
            traj.metrics["illegal_move"] = illegal_flag
            traj.metrics["turns"] = agent_turns
            traj.metrics["reward_outcome"] = base
            traj.metrics["reward_shaping"] = shaping_sum
            traj.reward = total
            return traj

