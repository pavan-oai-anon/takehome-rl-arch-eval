"""Rollout logic for Tic-Tac-Toe using ART framework."""

from typing import Any
from env import initialize_board, validate_move, check_winner, board_full, TRAINING_CONFIG, SYMBOLS
import art
import weave

@weave.op
@art.retry
async def rollout(model: art.Model, step: int, config: dict[str, Any]) -> art.Trajectory:
    board = initialize_board()
    current_player = 0  # 0 for 'X', 1 for 'O'
    moves = 0
    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": (
                    "You are playing Tic-Tac-Toe. Your goal is to win by marking a row, column, or diagonal with your symbol. "
                    "Please respond with a move in the format A1, B2, etc."
                ),
            }
        ],
        metadata={
            "board_state": "".join(str(cell or '.') for row in board for cell in row),
            "current_player": current_player
        },
        reward=0
    )
    client = art.local.LocalBackend().get_client()

    while True:
        trajectory.messages_and_choices.append(
            {"role": "user", "content": f"Current board:\n{board}"}
        )
        chat_completion = await client.chat.completions.create(
            model=model.name,
            messages=trajectory.messages(),
            max_completion_tokens=config.get('max_completion_tokens', 128)
        )

        move_content = chat_completion.choices[0].message.content
        row, col = ord(move_content[0].upper()) - 65, int(move_content[1]) - 1

        if not validate_move(board, row, col):
            trajectory.reward = -1  # Penalize illegal move
            trajectory.metadata['illegal_move'] = 1
            break

        board[row][col] = SYMBOLS[current_player]
        moves += 1

        winner = check_winner(board)
        if winner:
            trajectory.reward = 1 if winner == SYMBOLS[0] else -1
            trajectory.metadata['winner'] = winner
            break
        elif board_full(board):
            trajectory.reward = 0.1
            trajectory.metadata['drawn_game'] = 1
            break

        current_player = 1 - current_player

    trajectory.metadata['final_board'] = "".join(str(cell or '.') for row in board for cell in row)
    trajectory.metrics['total_moves'] = moves

    return trajectory

