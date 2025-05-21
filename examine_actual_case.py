#!/usr/bin/env python
"""
Examine how LPML-enhanced LLM makes decisions in a real Connect Four game.

This script:
1. Plays a single game against BabyPlayer
2. Renders the board at each step
3. Shows what LPML strategies are retrieved
4. Analyzes how the LLM incorporates this knowledge
5. Outputs a detailed analysis report in Markdown
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
import logging

# Add parent directory to path for imports
sys.path.append('../')

# Import Connect Four environment
from connect_four_gymnasium import ConnectFourEnv
from connect_four_gymnasium.players import BabyPlayer

# Import stable-baselines3 for PPO model
from stable_baselines3 import PPO

# Import OpenAI for LLM reasoning
from openai import OpenAI

# Import utils for vector database
from utils.xml_utils import search_vector_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def board_to_string(board: np.ndarray) -> str:
    """
    Convert a Connect Four board to a string representation.
    
    Args:
        board: 2D numpy array representing the board (6x7)
        
    Returns:
        String representation of the board
    """
    rows, cols = board.shape
    lines = []
    
    # Add column headers
    lines.append(' '.join(str(i) for i in range(cols)))
    
    # Add board rows
    for r in range(rows):
        row = []
        for c in range(cols):
            if board[r, c] == 0:
                row.append('.')
            elif board[r, c] == 1:
                row.append('X')
            else:
                row.append('O')
        lines.append(' '.join(row))
    
    return '\n'.join(lines)


def board_to_ascii_art(board: np.ndarray) -> str:
    """
    Convert a Connect Four board to an ASCII art representation.
    
    Args:
        board: 2D numpy array representing the board (6x7)
        
    Returns:
        ASCII art representation of the board
    """
    rows, cols = board.shape
    lines = []
    
    # Add top border
    lines.append('+' + '-' * (cols * 2 - 1) + '+')
    
    # Add board rows
    for r in range(rows):
        row = '|'
        for c in range(cols):
            if board[r, c] == 0:
                row += ' '
            elif board[r, c] == 1:
                row += 'X'
            else:
                row += 'O'
            
            if c < cols - 1:
                row += ' '
        
        row += '|'
        lines.append(row)
    
    # Add column numbers
    bottom = '|'
    for c in range(cols):
        bottom += str(c)
        if c < cols - 1:
            bottom += ' '
    bottom += '|'
    lines.append(bottom)
    
    # Add bottom border
    lines.append('+' + '-' * (cols * 2 - 1) + '+')
    
    return '\n'.join(lines)


class LPMLEnhancedLLMPlayer:
    """
    Connect Four player that uses an LLM enhanced with LPML annotations.
    
    This player retrieves relevant LPML annotations from a vector database
    based on the current board state and uses them to guide the LLM's
    decision-making process.
    """
    
    def __init__(
        self,
        vectordb_path: str,
        model_name: str = "gpt-4o-mini",
        name: str = "LPML-Enhanced LLM Player",
        temperature: float = 0.3,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the LPML-enhanced LLM player.
        
        Args:
            vectordb_path: Path to the vector database
            model_name: Name of the LLM model to use for reasoning
            name: Name of the player
            temperature: Temperature for LLM generation
            max_retries: Maximum number of retries for API calls
            retry_delay: Delay between retries in seconds
        """
        self.name = name
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize OpenAI client
        self.client = OpenAI()
        
        # Load vector database
        self.vectordb = search_vector_db(vectordb_path)
        
        # Check if vectordb is loaded
        if self.vectordb is None:
            raise ValueError("Vector database not loaded")
        
        # Store analysis for each move
        self.analysis_history = []
    
    def play(self, obs: np.ndarray) -> int:
        """
        Play a move based on the current observation.
        
        Args:
            obs: Current board observation
            
        Returns:
            Action to take (column index)
        """
        # Convert board to string for query
        board_str = board_to_string(obs)
        
        # Query vector database for relevant strategies
        results = self.vectordb.query(
            query_texts=[f"Board state: {board_str}"],
            n_results=3,
        )
        
        # Extract strategies
        if not results["documents"] or not results["documents"][0]:
            logger.warning("No relevant strategies found in vector database.")
            strategies = []
        else:
            strategies = results["documents"][0]
        
        # Store distances for analysis
        distances = []
        if "distances" in results and results["distances"] and results["distances"][0]:
            distances = results["distances"][0]
        
        # Build system prompt with LPML enhancement
        system_prompt = """
        You are a Connect Four strategy expert. You will analyze the current board state and determine the best move.
        
        You have been provided with some strategic knowledge from previous games as a reference, but you should make your own decision based on your analysis of the current board state.
        
        Consider the provided strategies but don't follow them blindly - they are just references to inspire your thinking.
        """
        
        # Build user prompt with LPML enhancement
        user_prompt = f"Current board state:\n{board_str}\n\n"
        
        if strategies:
            user_prompt += "Reference strategic knowledge from previous games:\n"
            
            for i, strategy in enumerate(strategies):
                similarity = "Unknown"
                if i < len(distances):
                    # Convert distance to similarity (1 - distance)
                    similarity = f"{(1 - distances[i]) * 100:.2f}%"
                
                user_prompt += f"Reference {i+1} (Similarity: {similarity}):\n{strategy}\n\n"
        else:
            user_prompt += "No relevant strategic knowledge found in the database. You'll need to rely entirely on your own analysis.\n\n"
        
        user_prompt += """
        Based on your own analysis of the current board state (with the above references as inspiration only), what is the best move?
        You are playing as 'X' and trying to connect four of your pieces.
        
        Respond in this format:
        
        Analysis: [Your own independent analysis of the board state]
        Consideration of references: [Brief thoughts on how the references informed your thinking]
        My decision: [Your reasoning for the final move choice]
        Best move: [column number, 0-6]
        
        Choose only one column number from 0 to 6 as your best move.
        """
        
        # Call OpenAI API with retries
        response_text = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=1000,
                )
                
                # Extract response text
                response_text = response.choices[0].message.content
                break
                
            except Exception as e:
                logger.warning(f"Error calling OpenAI API (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    # If all retries failed, raise the exception
                    raise
        
        # Parse response for action
        action = 3  # Default to center column
        
        if response_text:
            # Extract sections with their content
            sections = {}
            current_section = None
            
            for line in response_text.strip().split('\n'):
                line = line.strip()
                
                if line.startswith("Analysis:"):
                    current_section = "analysis"
                    sections[current_section] = line[len("Analysis:"):].strip()
                elif line.startswith("Consideration of references:"):
                    current_section = "consideration"
                    sections[current_section] = line[len("Consideration of references:"):].strip()
                elif line.startswith("My decision:"):
                    current_section = "decision"
                    sections[current_section] = line[len("My decision:"):].strip()
                elif line.startswith("Best move:"):
                    current_section = "best_move"
                    sections[current_section] = line[len("Best move:"):].strip()
                    
                    # Extract the action from this line
                    for char in line:
                        if char.isdigit() and int(char) >= 0 and int(char) <= 6:
                            action = int(char)
                            break
                elif current_section:
                    # Append to current section
                    sections[current_section] += " " + line
        
            # Store analysis for the report
            self.analysis_history.append({
                "board": obs.copy(),
                "board_str": board_str,
                "board_ascii": board_to_ascii_art(obs),
                "retrieved_strategies": strategies,
                "similarity_scores": distances,
                "llm_analysis": sections.get("analysis", ""),
                "consideration": sections.get("consideration", ""),
                "decision": sections.get("decision", ""),
                "action": action,
                "full_response": response_text
            })
        
        # Verify action is valid
        valid_actions = [c for c in range(7) if obs[0, c] == 0]
        if action not in valid_actions and valid_actions:
            action = np.random.choice(valid_actions)
            logger.warning(f"Invalid action {action}, choosing random valid action {action}")
        
        return action


def play_and_analyze_game(
    vectordb_path: str,
    llm_model_name: str = "gpt-4o-mini",
    output_dir: str = "results/analysis",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Play a single game with the LPML-enhanced LLM against BabyPlayer and analyze the decision-making process.
    
    Args:
        vectordb_path: Path to the vector database
        llm_model_name: Name of the LLM model to use
        output_dir: Directory to save results
        seed: Random seed
        
    Returns:
        Dictionary containing analysis results
    """
    # Set random seed
    np.random.seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize LPML-enhanced LLM player
    lpml_llm = LPMLEnhancedLLMPlayer(
        vectordb_path=vectordb_path,
        model_name=llm_model_name,
        name="LPML-Enhanced LLM",
    )
    
    # Initialize opponent
    opponent = BabyPlayer()
    
    # Create environment with rgb_array rendering (works on headless servers)
    env = ConnectFourEnv(opponent=opponent, render_mode="rgb_array")
    
    # Play game
    logger.info("Starting game: LPML-enhanced LLM vs BabyPlayer")
    obs, info = env.reset(seed=seed)
    
    # Render initial state
    logger.info("Initial board state:")
    
    # Game loop
    done = False
    move_count = 0
    turn_count = 0
    
    # Store all board states for analysis
    board_states = []
    
    # Capture initial board state
    board_states.append({
        "turn": turn_count,
        "player": "Initial",
        "board": obs.copy(),
        "action": None
    })
    turn_count += 1
    
    while not done:
        # Get action from LPML-enhanced LLM
        logger.info(f"Move {move_count + 1}: LPML-enhanced LLM thinking...")
        action = lpml_llm.play(obs)
        
        # Take step in environment
        logger.info(f"LPML-enhanced LLM plays column {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Store the board state after our move
        board_states.append({
            "turn": turn_count,
            "player": "LPML-LLM", 
            "board": obs.copy(),
            "action": action
        })
        turn_count += 1
        
        # Update game status
        done = terminated or truncated
        move_count += 1
        
        # If game is over, break
        if done:
            break
        
        # Opponent's move (automatically handled by the environment)
        # Store the board state after opponent's move
        board_states.append({
            "turn": turn_count,
            "player": "BabyPlayer",
            "board": obs.copy(),
            "action": info.get("opponent_action", None)
        })
        turn_count += 1
    
    # Determine result
    if reward > 0:
        result = "LPML-enhanced LLM wins"
    elif reward < 0:
        result = "BabyPlayer wins"
    else:
        result = "Draw"
    
    logger.info(f"Game result: {result}")
    
    # Generate report
    generate_analysis_report(
        lpml_llm.analysis_history,
        board_states=board_states,
        result=result,
        move_count=move_count,
        output_dir=output_dir,
    )
    
    return {
        "analysis_history": lpml_llm.analysis_history,
        "result": result,
        "move_count": move_count,
    }


def generate_analysis_report(
    analysis_history: List[Dict[str, Any]],
    board_states: List[Dict[str, Any]],
    result: str,
    move_count: int,
    output_dir: str,
) -> None:
    """
    Generate a detailed analysis report in Markdown format.
    
    Args:
        analysis_history: List of analysis data for each move
        result: Result of the game
        move_count: Total number of moves
        output_dir: Directory to save the report
    """
    # Create report path
    report_path = os.path.join(output_dir, "report.md")
    
    # Board images directory
    boards_dir = os.path.join(output_dir, "boards")
    os.makedirs(boards_dir, exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("# LPML-Enhanced LLM Decision Analysis\n\n")
        f.write(f"*Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        f.write("## Game Summary\n\n")
        f.write(f"- **Opponent**: BabyPlayer\n")
        f.write(f"- **Result**: {result}\n")
        f.write(f"- **Total Moves**: {move_count}\n\n")
        
        # First, generate board visualizations for all states in the game
        f.write("## Complete Game Visualization\n\n")
        
        for i, state in enumerate(board_states):
            board = state["board"]
            player = state["player"]
            action = state["action"]
            
            # Convert board to ASCII art
            board_ascii = board_to_ascii_art(board)
            
            # Draw board with matplotlib
            plt.figure(figsize=(7, 6))
            plt.imshow(board, cmap='coolwarm', vmin=-1, vmax=1)
            
            # Add grid
            for grid_i in range(board.shape[0] + 1):
                plt.axhline(grid_i - 0.5, color='black', linewidth=1)
            for grid_j in range(board.shape[1] + 1):
                plt.axvline(grid_j - 0.5, color='black', linewidth=1)
            
            # Add pieces
            for r in range(board.shape[0]):
                for c in range(board.shape[1]):
                    if board[r, c] == 1:
                        plt.text(c, r, 'X', ha='center', va='center', color='white', fontsize=18, fontweight='bold')
                    elif board[r, c] == -1:
                        plt.text(c, r, 'O', ha='center', va='center', color='black', fontsize=18, fontweight='bold')
            
            # Add column numbers at the bottom
            for c in range(board.shape[1]):
                plt.text(c, board.shape[0] - 0.5, str(c), ha='center', va='top', color='black', fontsize=12)
            
            # Set title based on player and action
            if player == "Initial":
                title = "Initial Board State"
            else:
                action_text = f" (Column {action})" if action is not None else ""
                title = f"Turn {i}: {player}{action_text}"
            
            plt.title(title)
            plt.axis('off')
            
            # Save the board image
            board_image_path = os.path.join(boards_dir, f"turn_{i}_board.png")
            plt.savefig(board_image_path)
            plt.close()
            
            # Add board state to report
            f.write(f"### {title}\n\n")
            f.write("```\n")
            f.write(board_ascii)
            f.write("\n```\n\n")
            f.write(f"![Board State](./boards/turn_{i}_board.png)\n\n")
        
        f.write("## LLM Decision Analysis\n\n")
        
        # Then, generate the detailed LLM analysis sections
        for i, move_data in enumerate(analysis_history):
            f.write(f"### Move {i+1}: Column {move_data['action']}\n\n")
            
            # Display board (already shown in the complete visualization)
            f.write("#### Board State Before Decision\n\n")
            f.write(f"![Board State](./boards/turn_{i*2}_board.png)\n\n")
            
            # Retrieved strategies
            f.write("#### Retrieved Strategic Knowledge\n\n")
            
            if not move_data["retrieved_strategies"]:
                f.write("*No relevant strategies found in the database.*\n\n")
            else:
                for j, (strategy, similarity) in enumerate(zip(
                        move_data["retrieved_strategies"], 
                        move_data["similarity_scores"] if move_data["similarity_scores"] else []
                    )):
                    similarity_str = f" (Similarity: {(1 - similarity) * 100:.2f}%)" if j < len(move_data["similarity_scores"]) else ""
                    f.write(f"**Reference {j+1}{similarity_str}:**\n\n")
                    f.write("```\n")
                    f.write(strategy)
                    f.write("\n```\n\n")
            
            # LLM analysis
            f.write("#### LLM Analysis\n\n")
            f.write(move_data["llm_analysis"])
            f.write("\n\n")
            
            # Consideration of references
            f.write("#### Consideration of References\n\n")
            f.write(move_data["consideration"])
            f.write("\n\n")
            
            # Decision
            f.write("#### Decision Process\n\n")
            f.write(move_data["decision"])
            f.write("\n\n")
            
            f.write("---\n\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        f.write("### Effectiveness of LPML Guidance\n\n")
        
        # Count moves with and without LPML guidance
        moves_with_lpml = sum(1 for move in analysis_history if move["retrieved_strategies"])
        moves_without_lpml = len(analysis_history) - moves_with_lpml
        
        f.write(f"- **Moves with LPML guidance**: {moves_with_lpml} ({moves_with_lpml / len(analysis_history) * 100:.1f}%)\n")
        f.write(f"- **Moves without LPML guidance**: {moves_without_lpml} ({moves_without_lpml / len(analysis_history) * 100:.1f}%)\n\n")
        
        # Final thoughts on LPML's impact
        if moves_with_lpml > moves_without_lpml:
            f.write("The LPML-enhanced LLM had strategic knowledge available for most of its moves. ")
        else:
            f.write("The LPML-enhanced LLM had to rely on its own analysis for most moves due to lack of relevant strategic knowledge. ")
        
        if result == "LPML-enhanced LLM wins":
            f.write("This knowledge appears to have been beneficial, as the LPML-enhanced LLM won the game.\n\n")
        elif result == "BabyPlayer wins":
            f.write("Despite this, the LPML-enhanced LLM lost the game, suggesting that either the strategic knowledge was not relevant enough or the LLM didn't incorporate it effectively.\n\n")
        else:
            f.write("The game ended in a draw, suggesting that the strategic knowledge had a neutral impact on the LLM's performance.\n\n")
        
        # Save raw analysis data
        with open(os.path.join(output_dir, "analysis_data.json"), "w") as json_file:
            # Convert numpy arrays to lists for JSON serialization
            serializable_history = []
            for move in analysis_history:
                serializable_move = move.copy()
                serializable_move["board"] = move["board"].tolist()
                serializable_history.append(serializable_move)
            
            json.dump(serializable_history, json_file, indent=2)
    
    logger.info(f"Analysis report generated at {report_path}")


def main():
    """Main function to run the analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze LPML-enhanced LLM decision-making in Connect Four")
    
    parser.add_argument("--vectordb", "-v", type=str, default="data/vectordb",
                       help="Path to the vector database (default: data/vectordb)")
    parser.add_argument("--model-name", "-m", type=str, default="gpt-4o-mini",
                       help="Name of the LLM model to use (default: gpt-4o-mini)")
    parser.add_argument("--output-dir", "-o", type=str, default="results/analysis",
                       help="Directory to save results (default: results/analysis)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                       help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    # Run analysis
    play_and_analyze_game(
        vectordb_path=args.vectordb,
        llm_model_name=args.model_name,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    
    logger.info(f"Analysis completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
