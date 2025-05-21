#!/usr/bin/env python
"""
Board encoder module for Connect Four.

This module provides functions to convert Connect Four board states
to human-readable text representations for LLM processing.
"""

import numpy as np
from typing import Union, List, Optional


def encode_board(board: Union[np.ndarray, List[List[int]]]) -> str:
    """
    Convert a Connect Four board state to a human-readable text representation.
    
    Args:
        board: The board state as a numpy array or nested list.
              Typically 6x7 with values:
              1 for player pieces
              -1 for opponent pieces
              0 for empty spaces
    
    Returns:
        A string representation of the board
    """
    # Convert to numpy array if it's a list
    if isinstance(board, list):
        board = np.array(board)
    
    # Ensure the board is 2D
    if board.ndim > 2:
        # If this is an observation with multiple channels, extract just the board state
        if board.ndim == 3 and board.shape[0] == 2:
            # Common format: [player_pieces, opponent_pieces] channels
            player_pieces = board[0]
            opponent_pieces = board[1]
            # Combine into a single board: 1 for player, -1 for opponent, 0 for empty
            board = np.zeros(player_pieces.shape, dtype=np.int8)
            board[player_pieces > 0] = 1
            board[opponent_pieces > 0] = -1
        else:
            # Try to flatten or extract the main board from complex state
            try:
                board = board.reshape(6, 7)
            except ValueError:
                # If we can't reshape, take the first valid board-like structure
                for i in range(board.shape[0]):
                    if board[i].shape == (6, 7) or board[i].shape == (7, 6):
                        board = board[i]
                        break
    
    # Ensure board is in the standard format (6 rows, 7 columns)
    rows, cols = board.shape
    if rows == 7 and cols == 6:
        # Transpose if needed
        board = board.T
        rows, cols = board.shape
    
    # If dimensions are still not standard, make a note
    dimensions_note = ""
    if rows != 6 or cols != 7:
        dimensions_note = f" (non-standard dimensions: {rows}x{cols})"
    
    # Create text representation
    symbols = {
        0: '⚪',  # Empty
        1: '🔴',  # Player
        -1: '🔵',  # Opponent
    }
    
    # Generate column headers
    col_headers = ' '.join([str(i) for i in range(cols)])
    
    # Generate board rows
    board_rows = []
    for r in range(rows):
        row = []
        for c in range(cols):
            value = int(board[r, c])
            # Use symbols for common values, otherwise use the numeric value
            symbol = symbols.get(value, str(value))
            row.append(symbol)
        board_rows.append(' '.join(row))
    
    # Create final board representation
    board_text = f"Connect Four Board{dimensions_note}:\n"
    board_text += f"{col_headers}\n"
    board_text += '\n'.join(board_rows)
    board_text += "\nColumn numbers: " + col_headers
    
    return board_text


def encode_board_ascii(board: Union[np.ndarray, List[List[int]]]) -> str:
    """
    Convert a Connect Four board state to an ASCII text representation.
    
    Args:
        board: The board state as a numpy array or nested list
    
    Returns:
        An ASCII string representation of the board
    """
    # Convert to numpy array if it's a list
    if isinstance(board, list):
        board = np.array(board)
    
    # Handle different board formats
    if board.ndim > 2:
        # If this is an observation with multiple channels, extract just the board state
        if board.ndim == 3 and board.shape[0] == 2:
            player_pieces = board[0]
            opponent_pieces = board[1]
            # Combine into a single board: 1 for player, -1 for opponent, 0 for empty
            board = np.zeros(player_pieces.shape, dtype=np.int8)
            board[player_pieces > 0] = 1
            board[opponent_pieces > 0] = -1
        else:
            # Try to flatten or extract the main board from complex state
            try:
                board = board.reshape(6, 7)
            except ValueError:
                # If we can't reshape, take the first valid board-like structure
                for i in range(board.shape[0]):
                    if board[i].shape == (6, 7) or board[i].shape == (7, 6):
                        board = board[i]
                        break
    
    # Ensure board is in the standard format (6 rows, 7 columns)
    rows, cols = board.shape
    if rows == 7 and cols == 6:
        # Transpose if needed
        board = board.T
        rows, cols = board.shape
    
    # Create ASCII representation
    symbols = {
        0: '.',  # Empty
        1: 'X',  # Player
        -1: 'O',  # Opponent
    }
    
    # Generate column headers
    col_headers = ' '.join([str(i) for i in range(cols)])
    
    # Generate board rows
    board_rows = []
    for r in range(rows):
        row = []
        for c in range(cols):
            value = int(board[r, c])
            # Use symbols for common values, otherwise use the numeric value
            symbol = symbols.get(value, str(value))
            row.append(symbol)
        board_rows.append(' '.join(row))
    
    # Create final board representation
    board_text = f"Connect Four Board:\n"
    board_text += f"{col_headers}\n"
    board_text += '\n'.join(board_rows)
    board_text += "\nLegend: X=Player, O=Opponent, .=Empty"
    
    return board_text


def board_to_text_description(
    board: Union[np.ndarray, List[List[int]]],
    turn_number: Optional[int] = None,
    player_symbol: str = "Red",
    opponent_symbol: str = "Yellow"
) -> str:
    """
    Generate a natural language description of the Connect Four board state.
    
    Args:
        board: The board state as a numpy array or nested list
        turn_number: Optional turn number to include in the description
        player_symbol: Symbol or color for the player
        opponent_symbol: Symbol or color for the opponent
        
    Returns:
        A natural language description of the board
    """
    # Convert to numpy array if it's a list
    if isinstance(board, list):
        board = np.array(board)
    
    # Handle different board formats (similar to encode_board)
    if board.ndim > 2:
        # If this is an observation with multiple channels, extract just the board state
        if board.ndim == 3 and board.shape[0] == 2:
            player_pieces = board[0]
            opponent_pieces = board[1]
            # Combine into a single board: 1 for player, -1 for opponent, 0 for empty
            board = np.zeros(player_pieces.shape, dtype=np.int8)
            board[player_pieces > 0] = 1
            board[opponent_pieces > 0] = -1
        else:
            # Try to flatten or extract the main board from complex state
            try:
                board = board.reshape(6, 7)
            except ValueError:
                # If we can't reshape, take the first valid board-like structure
                for i in range(board.shape[0]):
                    if board[i].shape == (6, 7) or board[i].shape == (7, 6):
                        board = board[i]
                        break
    
    # Ensure board is in the standard format (6 rows, 7 columns)
    rows, cols = board.shape
    if rows == 7 and cols == 6:
        # Transpose if needed
        board = board.T
        rows, cols = board.shape
    
    # Count pieces
    player_count = np.sum(board == 1)
    opponent_count = np.sum(board == -1)
    total_pieces = player_count + opponent_count
    
    # Determine whose turn it is based on piece count
    if player_count == opponent_count:
        current_turn = player_symbol
    else:
        current_turn = opponent_symbol
    
    # Start the description
    description = f"This is a Connect Four board with {total_pieces} pieces placed so far"
    if turn_number is not None:
        description += f", on turn {turn_number}"
    description += f". It is {current_turn}'s turn to play.\n\n"
    
    # Check for potential winning lines (3 in a row)
    player_threats = find_threats(board, 1)
    opponent_threats = find_threats(board, -1)
    
    if player_threats:
        description += f"{player_symbol} has potential winning moves at columns: {', '.join(map(str, player_threats))}.\n"
    
    if opponent_threats:
        description += f"{opponent_symbol} has potential winning moves at columns: {', '.join(map(str, opponent_threats))}.\n"
    
    # Add column-by-column description
    description += "\nColumn details:\n"
    for c in range(cols):
        column = board[:, c]
        empty_count = np.sum(column == 0)
        
        if empty_count == rows:
            description += f"- Column {c} is empty.\n"
            continue
            
        player_in_column = np.sum(column == 1)
        opponent_in_column = np.sum(column == -1)
        
        description += f"- Column {c} has {player_in_column} {player_symbol} pieces and {opponent_in_column} {opponent_symbol} pieces"
        
        # Check if column is full
        if empty_count == 0:
            description += " (full).\n"
        else:
            description += f" with {empty_count} empty spaces.\n"
    
    return description


def find_threats(board: np.ndarray, player: int) -> List[int]:
    """
    Find potential winning moves (three in a row with an empty space) for a player.
    
    Args:
        board: The board state as a numpy array
        player: Player value (1 or -1)
        
    Returns:
        List of column indices where the player has a potential winning move
    """
    rows, cols = board.shape
    threats = set()
    
    # Check for horizontal threats
    for r in range(rows):
        for c in range(cols - 3):
            window = board[r, c:c+4]
            if sum(window == player) == 3 and sum(window == 0) == 1:
                # Find the empty position
                for i in range(4):
                    if window[i] == 0 and (r == rows-1 or board[r+1, c+i] != 0):
                        threats.add(c + i)
    
    # Check for vertical threats
    for c in range(cols):
        for r in range(rows - 3):
            window = board[r:r+4, c]
            if sum(window == player) == 3 and sum(window == 0) == 1:
                # Find the empty position
                for i in range(4):
                    if window[i] == 0 and (r+i == rows-1 or board[r+i+1, c] != 0):
                        threats.add(c)
    
    # Check for diagonal threats (positive slope)
    for r in range(rows - 3):
        for c in range(cols - 3):
            window = [board[r+i, c+i] for i in range(4)]
            if sum(window == player) == 3 and sum(window == 0) == 1:
                # Find the empty position
                for i in range(4):
                    if window[i] == 0 and (r+i == rows-1 or board[r+i+1, c+i] != 0):
                        threats.add(c + i)
    
    # Check for diagonal threats (negative slope)
    for r in range(3, rows):
        for c in range(cols - 3):
            window = [board[r-i, c+i] for i in range(4)]
            if sum(window == player) == 3 and sum(window == 0) == 1:
                # Find the empty position
                for i in range(4):
                    if window[i] == 0 and (r-i == rows-1 or board[r-i+1, c+i] != 0):
                        threats.add(c + i)
    
    return sorted(list(threats))


if __name__ == "__main__":
    # Example usage
    example_board = np.zeros((6, 7), dtype=np.int8)
    example_board[5, 3] = 1  # Player piece in center bottom
    example_board[5, 4] = -1  # Opponent piece next to it
    example_board[4, 3] = 1  # Player piece stacked
    
    print(encode_board(example_board))
    print("\n" + "="*40 + "\n")
    print(encode_board_ascii(example_board))
    print("\n" + "="*40 + "\n")
    print(board_to_text_description(example_board, turn_number=3))
