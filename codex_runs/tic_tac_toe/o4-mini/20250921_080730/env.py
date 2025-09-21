"""Environment helpers for Tic-Tac-Toe ART RL setup.

This module provides constants, configuration, and utility functions.
"""
import random
import hashlib
from typing import List, Tuple, Optional

# Random seed for reproducibility
RANDOM_SEED: int = 42

# Board constants
EMPTY = " "
PLAYER_X = "X"
PLAYER_O = "O"
BOARD_SIZE = 3
ROWS = ["A", "B", "C"]
COLS = ["1", "2", "3"]

# Training configuration for ART
TRAINING_CONFIG = {
    "project": "tic_tac_toe",
    "model_name": "ttt-agent",
    "base_model": "gpt-3.5-turbo",
    "steps": 100,
    "trajectories_per_group": 10,
    "groups_per_step": 1,
    "learning_rate": 1e-5,
    "max_completion_tokens": 16,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_exceptions": 5,
    "cleanup_keep_last": 2,
}

def create_board() -> List[List[str]]:
    """Initialize an empty BOARD_SIZE x BOARD_SIZE board."""
    return [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

def render_board(board: List[List[str]]) -> str:
    """Return a text grid of the board."""
    lines: List[str] = []
    header = "  " + " ".join(COLS)
    lines.append(header)
    for i, row in enumerate(board):
        row_str = " ".join(cell if cell != EMPTY else "." for cell in row)
        lines.append(f"{ROWS[i]} {row_str}")
    return "\n".join(lines)

def parse_move(move: str) -> Optional[Tuple[int, int]]:
    """Parse a move like 'A1' into (row, col) indices, or None if invalid."""
    if len(move) != 2:
        return None
    row_char, col_char = move[0].upper(), move[1]
    if row_char not in ROWS or col_char not in COLS:
        return None
    return ROWS.index(row_char), COLS.index(col_char)

def is_valid_move(board: List[List[str]], move: str) -> bool:
    """Check if a move is on the board and the cell is empty."""
    pos = parse_move(move)
    if pos is None:
        return False
    r, c = pos
    return board[r][c] == EMPTY

def apply_move(board: List[List[str]], move: str, player: str) -> None:
    """Apply a valid move to the board, or raise ValueError."""
    pos = parse_move(move)
    if pos is None:
        raise ValueError("Invalid move format")
    r, c = pos
    if board[r][c] != EMPTY:
        raise ValueError("Cell already occupied")
    board[r][c] = player

def check_winner(board: List[List[str]]) -> Optional[str]:
    """Return PLAYER_X or PLAYER_O if there's a winner, 'D' for draw, else None."""
    lines = []
    # rows and columns
    lines.extend(board)
    lines.extend([[board[r][c] for r in range(BOARD_SIZE)] for c in range(BOARD_SIZE)])
    # diagonals
    lines.append([board[i][i] for i in range(BOARD_SIZE)])
    lines.append([board[i][BOARD_SIZE-1-i] for i in range(BOARD_SIZE)])
    for line in lines:
        if all(cell == PLAYER_X for cell in line):
            return PLAYER_X
        if all(cell == PLAYER_O for cell in line):
            return PLAYER_O
    # draw
    if all(cell != EMPTY for row in board for cell in row):
        return "D"
    return None

def board_hash(board: List[List[str]]) -> str:
    """Return an MD5 hash of the flattened board state."""
    flat = "".join(cell for row in board for cell in row)
    return hashlib.md5(flat.encode()).hexdigest()

# Seed randomness for reproducibility
random.seed(RANDOM_SEED)
