"""Configuration and environment helpers for Tic-Tac-Toe."""

import random
from typing import Dict

RANDOM_SEED = 42

TRAINING_CONFIG: Dict[str, any] = {
    'project': 'tic_tac_toe_rl',
    'model_name': 'ttt-agent',
    'base_model': 'Qwen/Qwen2.5-1.5B',
    'steps': 1000,
    'trajectories_per_group': 10,
    'groups_per_step': 2,
    'learning_rate': 1e-4,
    'max_completion_tokens': 128,
    'temperature': 0.7,
    'top_p': 0.9,
    'max_exceptions': 5,
    'cleanup_keep_last': 3,
}

BOARD_SIZE = 3
SYMBOLS = ['X', 'O']

def initialize_board() -> list[list[str | None]]:
    """Initialize an empty Tic-Tac-Toe board."""
    return [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

def validate_move(board: list[list[str | None]], row: int, col: int) -> bool:
    """Check if a move is valid for a given board state."""
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and board[row][col] is None

def check_winner(board: list[list[str | None]]) -> str | None:
    """Check for a winner on the board."""
    # Check rows
    for row in board:
        if row[0] is not None and all(symbol == row[0] for symbol in row):
            return row[0]
    # Check columns
    for col in range(BOARD_SIZE):
        if board[0][col] is not None and all(board[row][col] == board[0][col] for row in range(BOARD_SIZE)):
            return board[0][col]
    # Check diagonals
    if board[0][0] is not None and all(board[i][i] == board[0][0] for i in range(BOARD_SIZE)):
        return board[0][0]
    if board[0][BOARD_SIZE - 1] is not None and all(board[i][BOARD_SIZE - 1 - i] == board[0][BOARD_SIZE - 1] for i in range(BOARD_SIZE)):
        return board[0][BOARD_SIZE - 1]
    return None

def board_full(board: list[list[str | None]]) -> bool:
    """Check if the board is full."""
    return all(cell is not None for row in board for cell in row)

