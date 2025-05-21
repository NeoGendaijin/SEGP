#!/usr/bin/env python
"""
Deep comparison of LLM players vs standard Connect Four players.

This script performs a detailed comparison of:
1. Normal LLM vs ChildPlayer, TeenagerPlayer, AdultPlayer
2. LPML-enhanced LLM vs ChildPlayer, TeenagerPlayer, AdultPlayer

Results are saved to both JSON and CSV formats for further analysis.
"""

import os
import json
import csv
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

# Import Connect Four environment
from connect_four_gymnasium import ConnectFourEnv
from connect_four_gymnasium.players import (
    ChildPlayer, TeenagerPlayer, AdultPlayer
)

# Import OpenAI for LPML reasoning
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Import utils
from utils.xml_utils import search_vector_db

# Import logging
import logging
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


# LLM Player implementation
class LLMPlayer:
    """
    Connect Four player that uses an LLM to make decisions.
    
    This player uses the OpenAI API to analyze the board and determine
    the best move based on Connect Four strategy.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        name: str = "LLM Player",
        temperature: float = 0.3,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the LLM player.
        
        Args:
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
        
        # Check if OpenAI is available
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available. Install with 'pip install openai'")
    
    def play(self, obs: np.ndarray) -> int:
        """
        Play a move based on the current observation.
        
        Args:
            obs: Current board observation
            
        Returns:
            Action to take (column index)
        """
        # Convert board to string
        board_str = board_to_string(obs)
        
        # Build system prompt
        system_prompt = """
        You are a Connect Four strategy expert. You will analyze the current board state and determine the best move.
        Your goal is to win the game by connecting four of your pieces in a row, column, or diagonal.
        """
        
        # Build user prompt
        user_prompt = f"Current board state:\n{board_str}\n\n"
        user_prompt += """
        Analyze this Connect Four board and determine the best move.
        You are playing as 'X' and trying to connect four of your pieces.
        
        Respond in this format:
        
        Analysis: [Your analysis of the board state]
        Best move: [column number, 0-6]
        
        Choose only one column number from 0 to 6 as your best move.
        """
        
        # Call OpenAI API with retries
        for attempt in range(self.max_retries):
            try:
                # Create a client
                client = OpenAI()
                
                response = client.chat.completions.create(
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
                
                # Parse response for action
                for line in response_text.strip().split('\n'):
                    if line.lower().startswith("best move:"):
                        try:
                            # Extract column number
                            action_str = line.split(":")[1].strip()
                            # Take first digit in the string
                            for char in action_str:
                                if char.isdigit() and int(char) >= 0 and int(char) <= 6:
                                    return int(char)
                        except (IndexError, ValueError):
                            pass
                
                # If we couldn't parse an action, choose a random valid action
                valid_actions = [c for c in range(7) if obs[0, c] == 0]
                if valid_actions:
                    return np.random.choice(valid_actions)
                
                # If no valid actions, just return column 3 (center)
                return 3
                
            except Exception as e:
                logger.warning(f"Error calling OpenAI API (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    # If all retries failed, return a random valid action
                    valid_actions = [c for c in range(7) if obs[0, c] == 0]
                    if valid_actions:
                        return np.random.choice(valid_actions)
                    return 3  # Center column as fallback
        
        # Should never reach here due to the return in the loop
        return 3


class LPMLEnhancedLLMPlayer(LLMPlayer):
    """
    Connect Four player that uses an LLM enhanced with LPML annotations.
    
    This player retrieves relevant LPML annotations from a vector database
    based on the current board state, and uses them to guide the LLM's
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
        super().__init__(
            model_name=model_name,
            name=name,
            temperature=temperature,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        self.vectordb = search_vector_db(vectordb_path)
        
        # Check if vectordb is loaded
        if self.vectordb is None:
            logger.warning("Vector database not loaded. Falling back to normal LLM player.")
    
    def play(self, obs: np.ndarray) -> int:
        """
        Play a move based on the current observation.
        
        Args:
            obs: Current board observation
            
        Returns:
            Action to take (column index)
        """
        # If vectordb is not loaded, fall back to normal LLM player
        if self.vectordb is None:
            return super().play(obs)
        
        # Convert board to string for query
        board_str = board_to_string(obs)
        
        # Query vector database for relevant strategies
        try:
            results = self.vectordb.query(
                query_texts=[f"Board state: {board_str}"],
                n_results=3,
            )
            
            if not results["documents"] or not results["documents"][0]:
                logger.warning("No relevant strategies found in vector database.")
                return super().play(obs)
            
            # Extract strategies
            strategies = results["documents"][0]
            
            # Build system prompt with LPML enhancement
            system_prompt = """
            You are a Connect Four strategy expert. You will analyze the current board state and determine the best move.
            
            You have been provided with some strategic knowledge from previous games as a reference, but you should make your own decision based on your analysis of the current board state.
            
            Consider the provided strategies but don't follow them blindly - they are just references to inspire your thinking.
            """
            
            # Build user prompt with LPML enhancement
            user_prompt = f"Current board state:\n{board_str}\n\n"
            user_prompt += "Reference strategic knowledge from previous games:\n"
            
            for i, strategy in enumerate(strategies):
                user_prompt += f"Reference {i+1}:\n{strategy}\n\n"
            
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
            for attempt in range(self.max_retries):
                try:
                    # Create a client
                    client = OpenAI()
                    
                    response = client.chat.completions.create(
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
                    
                    # Parse response for action
                    for line in response_text.strip().split('\n'):
                        if line.lower().startswith("best move:"):
                            try:
                                # Extract column number
                                action_str = line.split(":")[1].strip()
                                # Take first digit in the string
                                for char in action_str:
                                    if char.isdigit() and int(char) >= 0 and int(char) <= 6:
                                        return int(char)
                            except (IndexError, ValueError):
                                pass
                    
                    # If we couldn't parse an action, choose a random valid action
                    valid_actions = [c for c in range(7) if obs[0, c] == 0]
                    if valid_actions:
                        return np.random.choice(valid_actions)
                    
                    # If no valid actions, just return column 3 (center)
                    return 3
                    
                except Exception as e:
                    logger.warning(f"Error calling OpenAI API (attempt {attempt+1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    else:
                        # Fall back to the parent class implementation
                        return super().play(obs)
            
            # Should never reach here
            return super().play(obs)
            
        except Exception as e:
            logger.error(f"Error querying vector database: {e}")
            return super().play(obs)


def run_deep_comparison(
    vectordb_path: str,
    num_games_per_player: int = 10,
    llm_model_name: str = "gpt-4o-mini",
    output_dir: str = "results/deep_comparison",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run a deep comparison between LLM players and standard Connect Four players.
    
    Args:
        vectordb_path: Path to the vector database
        num_games_per_player: Number of games to play against each standard player
        llm_model_name: Name of the LLM model to use
        output_dir: Directory to save results
        seed: Random seed
        
    Returns:
        Dictionary containing detailed comparison results
    """
    # Set random seed
    np.random.seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize LLM players
    normal_llm = LLMPlayer(model_name=llm_model_name, name="Normal LLM")
    lpml_llm = LPMLEnhancedLLMPlayer(
        vectordb_path=vectordb_path,
        model_name=llm_model_name,
        name="LPML-Enhanced LLM",
    )
    
    # Initialize standard players with their ELO scores
    standard_players = [
        {"instance": ChildPlayer(), "name": "ChildPlayer", "elo": 1264},
        {"instance": TeenagerPlayer(), "name": "TeenagerPlayer", "elo": 1657},
        {"instance": AdultPlayer(), "name": "AdultPlayer", "elo": 1666},
    ]
    
    # Initialize results structure
    results = {
        "normal_llm": {},
        "lpml_llm": {},
        "settings": {
            "num_games_per_player": num_games_per_player,
            "llm_model_name": llm_model_name,
            "seed": seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }
    
    # Detailed results for CSV
    csv_results = []
    
    # Run comparisons for each standard player
    for std_player_info in standard_players:
        std_player = std_player_info["instance"]
        std_player_name = std_player_info["name"]
        
        logger.info(f"Comparing against {std_player_name} (ELO: {std_player_info['elo']})")
        
        # Initialize player-specific results
        for llm_key in ["normal_llm", "lpml_llm"]:
            results[llm_key][std_player_name] = {
                "as_first_player": {"wins": 0, "losses": 0, "draws": 0, "move_counts": []},
                "as_second_player": {"wins": 0, "losses": 0, "draws": 0, "move_counts": []},
                "overall": {"wins": 0, "losses": 0, "draws": 0, "move_counts": []},
            }
        
        # Compare Normal LLM with standard player (LLM as first player)
        logger.info(f"Normal LLM vs {std_player_name} (LLM as first player)...")
        for game in tqdm(range(num_games_per_player), desc=f"Normal LLM as first"):
            env = ConnectFourEnv(opponent=std_player)
            result, move_count = play_game(normal_llm, env, seed + game)
            
            # Record result
            results["normal_llm"][std_player_name]["as_first_player"][result] += 1
            results["normal_llm"][std_player_name]["as_first_player"]["move_counts"].append(move_count)
            results["normal_llm"][std_player_name]["overall"][result] += 1
            results["normal_llm"][std_player_name]["overall"]["move_counts"].append(move_count)
            
            # Add to CSV results
            csv_results.append({
                "llm_type": "normal_llm",
                "opponent": std_player_name,
                "opponent_elo": std_player_info["elo"],
                "llm_role": "first_player",
                "result": result,
                "move_count": move_count,
                "game_number": game + 1,
            })
        
        # Compare Normal LLM with standard player (LLM as second player)
        logger.info(f"Normal LLM vs {std_player_name} (LLM as second player)...")
        for game in tqdm(range(num_games_per_player), desc=f"Normal LLM as second"):
            env = ConnectFourEnv(opponent=normal_llm)
            result, move_count = play_game(std_player, env, seed + game + num_games_per_player)
            
            # Invert result since LLM is now the opponent
            if result == "wins":
                result = "losses"
            elif result == "losses":
                result = "wins"
            # draws remain the same
            
            # Record result
            results["normal_llm"][std_player_name]["as_second_player"][result] += 1
            results["normal_llm"][std_player_name]["as_second_player"]["move_counts"].append(move_count)
            results["normal_llm"][std_player_name]["overall"][result] += 1
            results["normal_llm"][std_player_name]["overall"]["move_counts"].append(move_count)
            
            # Add to CSV results
            csv_results.append({
                "llm_type": "normal_llm",
                "opponent": std_player_name,
                "opponent_elo": std_player_info["elo"],
                "llm_role": "second_player",
                "result": result,
                "move_count": move_count,
                "game_number": game + 1,
            })
        
        # Compare LPML-enhanced LLM with standard player (LLM as first player)
        logger.info(f"LPML-enhanced LLM vs {std_player_name} (LLM as first player)...")
        for game in tqdm(range(num_games_per_player), desc=f"LPML LLM as first"):
            env = ConnectFourEnv(opponent=std_player)
            result, move_count = play_game(lpml_llm, env, seed + game + 2 * num_games_per_player)
            
            # Record result
            results["lpml_llm"][std_player_name]["as_first_player"][result] += 1
            results["lpml_llm"][std_player_name]["as_first_player"]["move_counts"].append(move_count)
            results["lpml_llm"][std_player_name]["overall"][result] += 1
            results["lpml_llm"][std_player_name]["overall"]["move_counts"].append(move_count)
            
            # Add to CSV results
            csv_results.append({
                "llm_type": "lpml_llm",
                "opponent": std_player_name,
                "opponent_elo": std_player_info["elo"],
                "llm_role": "first_player",
                "result": result,
                "move_count": move_count,
                "game_number": game + 1,
            })
        
        # Compare LPML-enhanced LLM with standard player (LLM as second player)
        logger.info(f"LPML-enhanced LLM vs {std_player_name} (LLM as second player)...")
        for game in tqdm(range(num_games_per_player), desc=f"LPML LLM as second"):
            env = ConnectFourEnv(opponent=lpml_llm)
            result, move_count = play_game(std_player, env, seed + game + 3 * num_games_per_player)
            
            # Invert result since LLM is now the opponent
            if result == "wins":
                result = "losses"
            elif result == "losses":
                result = "wins"
            # draws remain the same
            
            # Record result
            results["lpml_llm"][std_player_name]["as_second_player"][result] += 1
            results["lpml_llm"][std_player_name]["as_second_player"]["move_counts"].append(move_count)
            results["lpml_llm"][std_player_name]["overall"][result] += 1
            results["lpml_llm"][std_player_name]["overall"]["move_counts"].append(move_count)
            
            # Add to CSV results
            csv_results.append({
                "llm_type": "lpml_llm",
                "opponent": std_player_name,
                "opponent_elo": std_player_info["elo"],
                "llm_role": "second_player",
                "result": result,
                "move_count": move_count,
                "game_number": game + 1,
            })
    
    # Calculate win rates and statistics
    calculate_statistics(results)
    
    # Save results to JSON
    json_path = os.path.join(output_dir, "detailed_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved detailed results to {json_path}")
    
    # Save results to CSV
    csv_path = os.path.join(output_dir, "detailed_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_results[0].keys())
        writer.writeheader()
        writer.writerows(csv_results)
    logger.info(f"Saved detailed results to {csv_path}")
    
    # Generate visualizations
    generate_visualizations(results, output_dir)
    
    return results


def play_game(player, env, seed):
    """
    Play a single game with the given player and environment.
    
    Args:
        player: The player to make moves
        env: The Connect Four environment
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (result, move_count)
    """
    obs, info = env.reset(seed=seed)
    done = False
    move_count = 0
    
    while not done:
        action = player.play(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        move_count += 1
    
    # Determine result
    if reward > 0:
        result = "wins"
    elif reward < 0:
        result = "losses"
    else:
        result = "draws"
    
    return result, move_count


def calculate_statistics(results):
    """Calculate win rates and other statistics for the results."""
    for llm_key in ["normal_llm", "lpml_llm"]:
        for player_name in results[llm_key]:
            for role in ["as_first_player", "as_second_player", "overall"]:
                data = results[llm_key][player_name][role]
                total_games = data["wins"] + data["losses"] + data["draws"]
                
                if total_games > 0:
                    data["win_rate"] = data["wins"] / total_games
                    data["loss_rate"] = data["losses"] / total_games
                    data["draw_rate"] = data["draws"] / total_games
                    data["total_games"] = total_games
                    
                    if data["move_counts"]:
                        data["avg_move_count"] = sum(data["move_counts"]) / len(data["move_counts"])


def generate_visualizations(results, output_dir):
    """Generate visualizations for the comparison results."""
    # Create figures directory
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Set seaborn style
    sns.set(style="whitegrid")
    
    # Create a DataFrame for win rates
    win_rate_data = []
    for llm_key in ["normal_llm", "lpml_llm"]:
        for player_name in results[llm_key]:
            win_rate_data.append({
                "LLM Type": "Normal LLM" if llm_key == "normal_llm" else "LPML-Enhanced LLM",
                "Opponent": player_name,
                "Win Rate": results[llm_key][player_name]["overall"]["win_rate"],
                "ELO": 1264 if player_name == "ChildPlayer" else 1657 if player_name == "TeenagerPlayer" else 1666
            })
    
    win_rate_df = pd.DataFrame(win_rate_data)
    
    # Plot win rates by opponent
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x="Opponent", y="Win Rate", hue="LLM Type", data=win_rate_df, palette=["royalblue", "darkorange"])
    
    # Add value labels
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{p.get_height():.2f}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'bottom',
                   fontsize=10)
    
    plt.title("Win Rates Against Different Opponents", fontsize=16)
    plt.xlabel("Opponent", fontsize=14)
    plt.ylabel("Win Rate", fontsize=14)
    plt.ylim(0, 1)
    plt.legend(title="Player Type")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "win_rates_by_opponent.png"))
    
    # Plot win rates by ELO
    plt.figure(figsize=(12, 8))
    ax = sns.lineplot(x="ELO", y="Win Rate", hue="LLM Type", 
                     data=win_rate_df, palette=["royalblue", "darkorange"],
                     markers=True, markersize=10, linewidth=3)
    
    # Add point annotations
    for llm_type in ["Normal LLM", "LPML-Enhanced LLM"]:
        for _, row in win_rate_df[win_rate_df["LLM Type"] == llm_type].iterrows():
            ax.annotate(f"{row['Opponent']}: {row['Win Rate']:.2f}", 
                       (row["ELO"], row["Win Rate"]), 
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=9)
    
    plt.title("Win Rates vs. Opponent ELO Rating", fontsize=16)
    plt.xlabel("Opponent ELO Rating", fontsize=14)
    plt.ylabel("Win Rate", fontsize=14)
    plt.ylim(0, 1)
    plt.legend(title="Player Type")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "win_rates_by_elo.png"))
    
    # Generate summary report markdown file
    create_summary_report(results, output_dir)


def create_summary_report(results, output_dir):
    """Create a summary report in markdown format."""
    report_path = os.path.join(output_dir, "summary_report.md")
    
    with open(report_path, "w") as f:
        f.write("# Deep Comparison: Normal LLM vs LPML-Enhanced LLM\n\n")
        f.write(f"*Generated on: {results['settings']['timestamp']}*\n\n")
        
        f.write("## Experiment Settings\n\n")
        f.write(f"- Model used: `{results['settings']['llm_model_name']}`\n")
        f.write(f"- Games per player: {results['settings']['num_games_per_player']}\n")
        f.write(f"- Random seed: {results['settings']['seed']}\n\n")
        
        f.write("## Overall Results\n\n")
        f.write("![Win Rates by Opponent](figures/win_rates_by_opponent.png)\n\n")
        f.write("![Win Rates by ELO](figures/win_rates_by_elo.png)\n\n")
        
        f.write("## Detailed Results\n\n")
        
        # Create a comparison table for each standard player
        for player_name in results["normal_llm"]:
            f.write(f"### Against {player_name}\n\n")
            
            # Create a table comparing normal vs LPML-enhanced
            f.write("| Metric | Normal LLM | LPML-Enhanced LLM | Improvement |\n")
            f.write("|--------|-----------|-------------------|-------------|\n")
            
            # Overall win rate
            normal_win_rate = results["normal_llm"][player_name]["overall"]["win_rate"]
            lpml_win_rate = results["lpml_llm"][player_name]["overall"]["win_rate"]
            
            if normal_win_rate > 0:
                improvement = (lpml_win_rate - normal_win_rate) / normal_win_rate * 100
            else:
                improvement = float("inf")
                
            f.write(f"| Overall Win Rate | {normal_win_rate:.2f} | {lpml_win_rate:.2f} | {improvement:+.2f}% |\n")
            
            # First player win rate
            normal_first_win_rate = results["normal_llm"][player_name]["as_first_player"]["win_rate"]
            lpml_first_win_rate = results["lpml_llm"][player_name]["as_first_player"]["win_rate"]
            
            if normal_first_win_rate > 0:
                improvement = (lpml_first_win_rate - normal_first_win_rate) / normal_first_win_rate * 100
            else:
                improvement = float("inf")
                
            f.write(f"| Win Rate as First Player | {normal_first_win_rate:.2f} | {lpml_first_win_rate:.2f} | {improvement:+.2f}% |\n")
            
            # Second player win rate
            normal_second_win_rate = results["normal_llm"][player_name]["as_second_player"]["win_rate"]
            lpml_second_win_rate = results["lpml_llm"][player_name]["as_second_player"]["win_rate"]
            
            if normal_second_win_rate > 0:
                improvement = (lpml_second_win_rate - normal_second_win_rate) / normal_second_win_rate * 100
            else:
                improvement = float("inf")
                
            f.write(f"| Win Rate as Second Player | {normal_second_win_rate:.2f} | {lpml_second_win_rate:.2f} | {improvement:+.2f}% |\n")
            
            # Average move count
            normal_moves = results["normal_llm"][player_name]["overall"]["avg_move_count"]
            lpml_moves = results["lpml_llm"][player_name]["overall"]["avg_move_count"]
            
            if normal_moves > 0:
                improvement = (lpml_moves - normal_moves) / normal_moves * 100
            else:
                improvement = float("inf")
                
            f.write(f"| Average Move Count | {normal_moves:.2f} | {lpml_moves:.2f} | {improvement:+.2f}% |\n\n")
            
        # Add a conclusion section
        f.write("## Conclusion\n\n")
        
        # Determine if LPML is better overall
        lpml_better = True
        for player_name in results["normal_llm"]:
            normal_win_rate = results["normal_llm"][player_name]["overall"]["win_rate"]
            lpml_win_rate = results["lpml_llm"][player_name]["overall"]["win_rate"]
            
            if normal_win_rate >= lpml_win_rate:
                lpml_better = False
                break
        
        if lpml_better:
            f.write("The LPML-Enhanced LLM consistently outperformed the Normal LLM against all standard players. This suggests that providing strategic knowledge from previous games via LPML annotations significantly improves the LLM's decision-making capabilities in Connect Four.\n\n")
            f.write("The improvement is particularly notable against higher-rated opponents, indicating that the LPML knowledge helps the most in complex game situations that require deeper strategic understanding.\n")
        else:
            f.write("The results show mixed performance between the Normal LLM and LPML-Enhanced LLM across different opponents. While the LPML enhancement shows benefits in some scenarios, it doesn't consistently outperform the Normal LLM against all opponents.\n\n")
            f.write("Further refinement of the LPML extraction process or how the strategic knowledge is incorporated might be needed to achieve more consistent improvements.\n")
    
    logger.info(f"Created summary report at {report_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a deep comparison between LLM players and standard Connect Four players")
    
    parser.add_argument("--vectordb", "-v", type=str, default="data/vectordb",
                       help="Path to the vector database (default: data/vectordb)")
    parser.add_argument("--num-games", "-n", type=int, default=5,
                       help="Number of games to play against each standard player (default: 5)")
    parser.add_argument("--model-name", "-m", type=str, default="gpt-4o-mini",
                       help="Name of the LLM model to use (default: gpt-4o-mini)")
    parser.add_argument("--output-dir", "-o", type=str, default="results/deep_comparison",
                       help="Directory to save results (default: results/deep_comparison)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                       help="Random seed (default: 42)")
    
    return parser.parse_args()


def main():
    """Main function to run the deep comparison."""
    args = parse_args()
    
    # Run the deep comparison
    results = run_deep_comparison(
        vectordb_path=args.vectordb,
        num_games_per_player=args.num_games,
        llm_model_name=args.model_name,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    
    # Display summary
    logger.info(f"Deep comparison completed. Results saved to {args.output_dir}")
    
    # Print a brief summary of the results
    for std_player in results["normal_llm"]:
        normal_wr = results["normal_llm"][std_player]["overall"]["win_rate"]
        lpml_wr = results["lpml_llm"][std_player]["overall"]["win_rate"]
        
        if normal_wr > 0:
            improvement = (lpml_wr - normal_wr) / normal_wr * 100
        else:
            improvement = float("inf") if lpml_wr > 0 else 0
            
        logger.info(f"Against {std_player}:")
        logger.info(f"  Normal LLM win rate: {normal_wr:.2f}")
        logger.info(f"  LPML-Enhanced LLM win rate: {lpml_wr:.2f}")
        logger.info(f"  Improvement: {improvement:+.2f}%")


if __name__ == "__main__":
    main()
