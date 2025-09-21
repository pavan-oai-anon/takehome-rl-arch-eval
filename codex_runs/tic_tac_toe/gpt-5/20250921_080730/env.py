"""Tic-Tac-Toe helpers and shared configuration for ART rollouts.

This module exposes small, composable utilities used by rollout.py.
All hyperparameters and toggles live at the top for easy tweaking.

Notes
- We assume a LocalBackend for training/inference.
- If you need to install dependencies, prefer uv, e.g.:
  uv pip install openai weave  # plus your ART package
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple


# -----------------------------
# Tunable environment constants
# -----------------------------
BOARD_SIZE: int = 3
AGENT_MARK: str = "X"  # Agent plays as X
OPP_MARK: str = "O"    # Built-in opponent plays as O
EMPTY: str = "."
ROW_LABELS: str = "ABC"
COL_LABELS: str = "123"
RANDOM_SEED: int = 17

# Training config consumed by the host training loop.
# Keep values modest for local runs; adjust as needed.
TRAINING_CONFIG: dict[str, Any] = {
    "project": "tic-tac-toe-rl",
    "model_name": "agent-ttt-001",
    "base_model": "Qwen/Qwen2.5-1.5B",  # Small for local; tune GPU memory like 2048 example if needed.
    "steps": 20,
    "trajectories_per_group": 16,
    "groups_per_step": 1,
    "learning_rate": 1e-5,
    "max_completion_tokens": 8,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_exceptions": 32,
    "cleanup_keep_last": 1,
}


# -----------------------------
# Game state and creation
# -----------------------------
@dataclass
class TTTState:
    """Simple Tic-Tac-Toe state container.

    - board uses `EMPTY`, `AGENT_MARK`, `OPP_MARK`
    - turn holds the mark that should play next
    - last_move is like "B2" or ""
    - id is a short unique token for tracing
    """

    board: list[list[str]]
    turn: str
    last_move: str
    id: str


def _new_id(k: int = 6) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=k))


def new_game(seed: Optional[int] = None) -> TTTState:
    """Create a fresh game with an empty board and agent to play first.

    Args:
        seed: optional RNG seed for deterministic IDs in tests.
    """
    if seed is not None:
        rnd_state = random.getstate()
        random.seed(seed)
        uid = _new_id()
        random.setstate(rnd_state)
    else:
        uid = _new_id()
    board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    return TTTState(board=board, turn=AGENT_MARK, last_move="", id=uid)


# -----------------------------
# Rendering and parsing helpers
# -----------------------------
def board_to_text(s: TTTState) -> str:
    """Return a compact human- and model-friendly board string.

    Example:
        """
    header = "  " + " ".join(COL_LABELS[:BOARD_SIZE])
    rows = [
        f"{ROW_LABELS[r]} " + " ".join(s.board[r][c] for c in range(BOARD_SIZE))
        for r in range(BOARD_SIZE)
    ]
    return "\n".join([header, *rows])


def parse_move(move: str) -> Tuple[int, int]:
    """Parse a move like "B2" into 0-based (row, col).

    Accepts optional surrounding whitespace and lowercase.
    Raises ValueError on malformed inputs.
    """
    if not isinstance(move, str):
        raise ValueError("move must be a string like 'B2'")
    move = move.strip().upper()
    if len(move) != 2:
        raise ValueError("move must be two chars, e.g., B2")
    row_ch, col_ch = move[0], move[1]
    if row_ch not in ROW_LABELS[:BOARD_SIZE] or col_ch not in COL_LABELS[:BOARD_SIZE]:
        raise ValueError("move out of range")
    r = ROW_LABELS.index(row_ch)
    c = COL_LABELS.index(col_ch)
    return r, c


def is_legal_move(s: TTTState, move: str) -> bool:
    """Check if a coordinate refers to an empty square in bounds."""
    try:
        r, c = parse_move(move)
    except ValueError:
        return False
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and s.board[r][c] == EMPTY


def apply_move(s: TTTState, move: str, mark: Optional[str] = None) -> None:
    """Apply a legal move for the given mark and flip the turn.

    Raises ValueError if the move is illegal.
    """
    if mark is None:
        mark = s.turn
    if not is_legal_move(s, move):
        raise ValueError("illegal move")
    r, c = parse_move(move)
    s.board[r][c] = mark
    s.last_move = move.strip().upper()
    s.turn = OPP_MARK if mark == AGENT_MARK else AGENT_MARK


# -----------------------------
# Win/draw detection and hashing
# -----------------------------
def _lines(board: list[list[str]]) -> Iterable[list[Tuple[int, int]]]:
    # Rows and columns
    for i in range(BOARD_SIZE):
        yield [(i, j) for j in range(BOARD_SIZE)]
        yield [(j, i) for j in range(BOARD_SIZE)]
    # Diagonals
    yield [(i, i) for i in range(BOARD_SIZE)]
    yield [(i, BOARD_SIZE - 1 - i) for i in range(BOARD_SIZE)]


def check_winner(s: TTTState) -> str:
    """Return AGENT_MARK, OPP_MARK, or "" if no winner yet."""
    for line in _lines(s.board):
        marks = {s.board[r][c] for r, c in line}
        if len(marks) == 1:
            m = next(iter(marks))
            if m != EMPTY:
                return m
    return ""


def board_full(s: TTTState) -> bool:
    return all(cell != EMPTY for row in s.board for cell in row)


def outcome(s: TTTState) -> str:
    """Return "X", "O", or "draw" when terminal; else ""."""
    w = check_winner(s)
    if w:
        return w
    if board_full(s):
        return "draw"
    return ""


def board_hash(s: TTTState) -> str:
    """Stable compact serialization for metadata."""
    return "".join("".join(row) for row in s.board)


# -----------------------------
# Opponent policy and shaping
# -----------------------------
def _empty_squares(s: TTTState) -> list[str]:
    coords: list[str] = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if s.board[r][c] == EMPTY:
                coords.append(f"{ROW_LABELS[r]}{COL_LABELS[c]}")
    return coords


def _immediate_wins(board: list[list[str]], mark: str) -> list[Tuple[int, int]]:
    wins: list[Tuple[int, int]] = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != EMPTY:
                continue
            board[r][c] = mark
            if _is_win(board, mark):
                wins.append((r, c))
            board[r][c] = EMPTY
    return wins


def _is_win(board: list[list[str]], mark: str) -> bool:
    for line in _lines(board):
        if all(board[r][c] == mark for r, c in line):
            return True
    return False


def opponent_policy(s: TTTState, rnd: random.Random) -> str:
    """Reasonable baseline: win > block > center > corners > random."""
    # Try to win
    wins = _immediate_wins(s.board, OPP_MARK)
    if wins:
        r, c = wins[0]
        return f"{ROW_LABELS[r]}{COL_LABELS[c]}"
    # Try to block agent's win
    blocks = _immediate_wins(s.board, AGENT_MARK)
    if blocks:
        r, c = blocks[0]
        return f"{ROW_LABELS[r]}{COL_LABELS[c]}"
    # Center
    mid = BOARD_SIZE // 2
    if s.board[mid][mid] == EMPTY:
        return f"{ROW_LABELS[mid]}{COL_LABELS[mid]}"
    # Corners
    corners = [(0, 0), (0, BOARD_SIZE - 1), (BOARD_SIZE - 1, 0), (BOARD_SIZE - 1, BOARD_SIZE - 1)]
    rnd.shuffle(corners)
    for r, c in corners:
        if s.board[r][c] == EMPTY:
            return f"{ROW_LABELS[r]}{COL_LABELS[c]}"
    # Anything
    empties = _empty_squares(s)
    return rnd.choice(empties) if empties else f"{ROW_LABELS[0]}{COL_LABELS[0]}"


def count_immediate_threats(s: TTTState, mark: str) -> int:
    """Count placements that would win next turn for a mark."""
    return len(_immediate_wins(s.board, mark))


def shaping_after_agent_move(
    before: TTTState, after: TTTState, move: str
) -> float:
    """Compute small, smooth shaping after the agent's legal move.

    Heuristics (bounded to keep signal gentle):
    - +0.10 for center capture
    - +0.15 per new agent immediate threat (two-in-a-row with one empty)
    - +0.20 if the move reduces opponent immediate wins (i.e., blocks)
    """
    r, c = parse_move(move)
    reward = 0.0
    mid = BOARD_SIZE // 2
    if (r, c) == (mid, mid):
        reward += 0.10

    # Threat deltas
    agent_threats_before = count_immediate_threats(before, AGENT_MARK)
    opp_threats_before = count_immediate_threats(before, OPP_MARK)
    agent_threats_after = count_immediate_threats(after, AGENT_MARK)
    opp_threats_after = count_immediate_threats(after, OPP_MARK)

    if agent_threats_after > agent_threats_before:
        reward += 0.15 * (agent_threats_after - agent_threats_before)
    if opp_threats_after < opp_threats_before:
        reward += 0.20

    # Keep shaping small
    return max(-0.5, min(0.5, reward))


__all__ = [
    "RANDOM_SEED",
    "TRAINING_CONFIG",
    "TTTState",
    "AGENT_MARK",
    "OPP_MARK",
    "EMPTY",
    "new_game",
    "board_to_text",
    "parse_move",
    "is_legal_move",
    "apply_move",
    "check_winner",
    "board_full",
    "board_hash",
    "outcome",
    "opponent_policy",
    "shaping_after_agent_move",
]

