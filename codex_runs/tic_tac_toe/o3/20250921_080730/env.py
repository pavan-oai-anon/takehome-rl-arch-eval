"""Tic-Tac-Toe ART environment helpers.

This module contains all stateless helpers and hyper-parameters so that they
can be imported both by the rollout implementation and (optionally) by any
external evaluation scripts. Keep everything easily tweakable by editing the
constants below.
"""

from __future__ import annotations

import random
from typing import Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Global randomness & training configuration
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 2025

# Minimal configuration object consumed by OpenPipe's generic training loop.
# Feel free to change *values* – the keys themselves are required.
TRAINING_CONFIG: dict = {
    "project": "tic_tac_toe_rl",
    "model_name": "agent-ttt",
    "base_model": "Qwen/Qwen2.5-1.5B",
    "steps": 60,
    "trajectories_per_group": 16,
    "groups_per_step": 1,
    "learning_rate": 1e-5,
    "max_completion_tokens": 32,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_exceptions": 16,
    "cleanup_keep_last": 2,
}

# ---------------------------------------------------------------------------
# Game helpers
# ---------------------------------------------------------------------------

# Internally we treat the board as a 3x3 list of strings: "X", "O" or " " (blank).
Board = List[List[str]]

# Mapping from human-friendly move notation (e.g. "B2") to 0-based indices.
_ROW_LABELS = ("A", "B", "C")
_COL_LABELS = ("1", "2", "3")


def empty_board() -> Board:
    """Return a fresh 3 × 3 board filled with blanks."""

    return [[" " for _ in range(3)] for _ in range(3)]


def board_hash(board: Board) -> str:
    """Return a compact string hash of the board for logging purposes."""

    return "".join(cell if cell != " " else "_" for row in board for cell in row)


def render_board(board: Board) -> str:
    """Render *board* as a user-readable grid for the chat prompt.

    Example output::

        _ 1 2 3
        A X _ O
        B _ X _
        C _ _ O
    """

    header = "  " + " ".join(_COL_LABELS)
    rows = [
        f"{_ROW_LABELS[r]} " + " ".join(cell if cell != " " else "_" for cell in board[r])
        for r in range(3)
    ]
    return "\n".join([header] + rows)


def parse_move(move: str) -> Optional[Tuple[int, int]]:
    """Convert ``move`` like "B2" to 0-based row/col indices.

    Returns ``None`` if the string is malformed.
    """

    move = move.strip().upper()
    if len(move) != 2:
        return None
    row_c, col_c = move[0], move[1]
    if row_c not in _ROW_LABELS or col_c not in _COL_LABELS:
        return None
    return _ROW_LABELS.index(row_c), _COL_LABELS.index(col_c)


def check_winner(board: Board) -> Optional[str]:
    """Return "X" or "O" if a player has won, otherwise ``None``."""

    lines: Iterable[Iterable[Tuple[int, int]]] = (
        # Rows
        [(r, c) for c in range(3)] for r in range(3)
    )
    lines = list(lines) + [
        # Columns
        [(r, c) for r in range(3)] for c in range(3)
    ] + [
        # Diagonals
        [(0, 0), (1, 1), (2, 2)],
        [(0, 2), (1, 1), (2, 0)],
    ]

    for coords in lines:  # type: ignore[assignment]
        values = {board[r][c] for r, c in coords}
        if len(values) == 1:
            (value,) = values
            if value != " ":
                return value
    return None


def board_full(board: Board) -> bool:
    """Return True if no empty squares remain."""

    return all(cell != " " for row in board for cell in row)


def random_legal_move(board: Board) -> Tuple[int, int]:
    """Return random row/col for an empty square. Caller must ensure availability."""

    empties = [(r, c) for r in range(3) for c in range(3) if board[r][c] == " "]
    return random.choice(empties)


# Seed global RNG immediately so that unit tests & training are deterministic.
random.seed(RANDOM_SEED)

