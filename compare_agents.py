#!/usr/bin/env python
"""
Compare performance of Connect Four agents with different LLM enhancements.

This script compares the performance of:
1. Normal LLM agent vs LPML-enhanced LLM agent
2. Baby player vs Normal LLM agent
3. Various other combinations as requested

The LPML-enhanced LLM uses retrieved language-based strategic knowledge from
a vector database to guide its decision-making, implementing a Retrieval
Augmented Generation (RAG) approach.
"""

import os
import json
import pickle
import logging
import argparse
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Import stable-baselines3
from stable_baselines3 import PPO

# Import Connect Four environment
from connect_four_gymnasium import ConnectFourEnv
from connect_four_gymnasium.players import (
    BabyPlayer, ChildPlayer, TeenagerPlayer, AdultPlayer,
    ModelPlayer, ConsolePlayer
)

# Import OpenAI for LPML reasoning
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Import utils
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


# Create a custom player class instead of inheriting from Player
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


def run_comparison(
    vectordb_path: str,
    num_games: int = 20,
    opponent_pairs: List[Tuple[str, str]] = None,
    ppo_model_path: Optional[str] = None,
    llm_model_name: str = "gpt-4o-mini",
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Run a comparison between different pairs of agents.
    
    Args:
        vectordb_path: Path to the vector database
        num_games: Number of games to play for each pair
        opponent_pairs: List of pairs of opponents to compare
        ppo_model_path: Path to the trained PPO model (if needed)
        llm_model_name: Name of the LLM model to use
        seed: Random seed
        
    Returns:
        Dictionary containing comparison results
    """
    # Set random seed
    np.random.seed(seed)
    
    # Default opponent pairs if none provided
    if opponent_pairs is None:
        opponent_pairs = [
            ("normal_llm", "lpml_llm"),  # Normal LLM vs LPML-enhanced LLM
            ("baby", "normal_llm"),      # Baby player vs Normal LLM
        ]
    
    # Load PPO model if path provided and needed
    ppo_model = None
    if ppo_model_path and any(p in ["ppo"] for pair in opponent_pairs for p in pair):
        ppo_model = PPO.load(ppo_model_path)
        logger.info(f"Loaded PPO model from {ppo_model_path}")
    
    # Create player instances
    players = {}
    
    # Initialize results
    results = {}
    
    # Run comparison for each pair
    for pair in opponent_pairs:
        player1_type, player2_type = pair
        
        # Create player 1
        if player1_type not in players:
            players[player1_type] = create_player(
                player_type=player1_type,
                ppo_model=ppo_model,
                vectordb_path=vectordb_path,
                llm_model_name=llm_model_name,
            )
        
        # Create player 2
        if player2_type not in players:
            players[player2_type] = create_player(
                player_type=player2_type,
                ppo_model=ppo_model,
                vectordb_path=vectordb_path,
                llm_model_name=llm_model_name,
            )
        
        # Get player instances
        player1 = players[player1_type]
        player2 = players[player2_type]
        
        # Add name attribute to players that don't have it
        if not hasattr(player1, 'name'):
            player1.name = player1_type.capitalize() + " Player"
        if not hasattr(player2, 'name'):
            player2.name = player2_type.capitalize() + " Player"
        
        logger.info(f"Running comparison: {player1.name} vs {player2.name}")
        
        # Initialize results for this pair
        pair_key = f"{player1_type}_vs_{player2_type}"
        results[pair_key] = {
            "player1": {
                "type": player1_type,
                "name": player1.name,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "scores": [],
                "move_counts": []
            },
            "player2": {
                "type": player2_type,
                "name": player2.name,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "scores": [],
                "move_counts": []
            }
        }
        
        # Run games with player1 as the main player and player2 as the opponent
        logger.info(f"Running {num_games} games with {player1.name} as the main player")
        
        for game in tqdm(range(num_games), desc=f"{player1.name} as main"):
            # Create environment with player2 as the opponent
            env = ConnectFourEnv(opponent=player2)
            
            # Play game
            obs, info = env.reset(seed=seed + game)
            done = False
            move_count = 0
            
            while not done:
                # Get action from player1
                action = player1.play(obs)
                
                # Take step in environment
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Update done flag
                done = terminated or truncated
                move_count += 1
            
            # Update results
            if reward > 0:
                results[pair_key]["player1"]["wins"] += 1
                results[pair_key]["player1"]["scores"].append(1)
                results[pair_key]["player2"]["losses"] += 1
                results[pair_key]["player2"]["scores"].append(-1)
            elif reward < 0:
                results[pair_key]["player1"]["losses"] += 1
                results[pair_key]["player1"]["scores"].append(-1)
                results[pair_key]["player2"]["wins"] += 1
                results[pair_key]["player2"]["scores"].append(1)
            else:
                results[pair_key]["player1"]["draws"] += 1
                results[pair_key]["player1"]["scores"].append(0)
                results[pair_key]["player2"]["draws"] += 1
                results[pair_key]["player2"]["scores"].append(0)
            
            results[pair_key]["player1"]["move_counts"].append(move_count)
            results[pair_key]["player2"]["move_counts"].append(move_count)
        
        # Run games with player2 as the main player and player1 as the opponent
        logger.info(f"Running {num_games} games with {player2.name} as the main player")
        
        for game in tqdm(range(num_games), desc=f"{player2.name} as main"):
            # Create environment with player1 as the opponent
            env = ConnectFourEnv(opponent=player1)
            
            # Play game
            obs, info = env.reset(seed=seed + game + num_games)  # Different seeds
            done = False
            move_count = 0
            
            while not done:
                # Get action from player2
                action = player2.play(obs)
                
                # Take step in environment
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Update done flag
                done = terminated or truncated
                move_count += 1
            
            # Update results
            if reward > 0:
                results[pair_key]["player2"]["wins"] += 1
                results[pair_key]["player2"]["scores"].append(1)
                results[pair_key]["player1"]["losses"] += 1
                results[pair_key]["player1"]["scores"].append(-1)
            elif reward < 0:
                results[pair_key]["player2"]["losses"] += 1
                results[pair_key]["player2"]["scores"].append(-1)
                results[pair_key]["player1"]["wins"] += 1
                results[pair_key]["player1"]["scores"].append(1)
            else:
                results[pair_key]["player2"]["draws"] += 1
                results[pair_key]["player2"]["scores"].append(0)
                results[pair_key]["player1"]["draws"] += 1
                results[pair_key]["player1"]["scores"].append(0)
            
            results[pair_key]["player2"]["move_counts"].append(move_count)
            results[pair_key]["player1"]["move_counts"].append(move_count)
        
        # Calculate statistics
        for player_key in ["player1", "player2"]:
            player = results[pair_key][player_key]
            total_games = player["wins"] + player["losses"] + player["draws"]
            if total_games > 0:
                player["win_rate"] = player["wins"] / total_games
                player["avg_score"] = sum(player["scores"]) / total_games
                player["avg_moves"] = sum(player["move_counts"]) / total_games
        
        logger.info(f"Completed {pair_key}: {player1.name} win rate: {results[pair_key]['player1']['win_rate']:.2f}, "
                    f"{player2.name} win rate: {results[pair_key]['player2']['win_rate']:.2f}")
    
    return results


def create_player(
    player_type: str,
    ppo_model: Optional[Any] = None,
    vectordb_path: Optional[str] = None,
    llm_model_name: str = "gpt-4o-mini",
) -> Any:
    """
    Create a player based on the player type.
    
    Args:
        player_type: Type of player to create
        ppo_model: PPO model to use (if needed)
        vectordb_path: Path to the vector database (if needed)
        llm_model_name: Name of the LLM model to use (if needed)
        
    Returns:
        Player instance
    """
    if player_type == "baby":
        return BabyPlayer()
    elif player_type == "child":
        return ChildPlayer()
    elif player_type == "teenager":
        return TeenagerPlayer()
    elif player_type == "adult":
        return AdultPlayer()
    elif player_type == "ppo":
        if ppo_model is None:
            raise ValueError("PPO model required for player type 'ppo'")
        return ModelPlayer(ppo_model, name="PPO")
    elif player_type == "normal_llm":
        return LLMPlayer(model_name=llm_model_name, name="Normal LLM")
    elif player_type == "lpml_llm":
        if vectordb_path is None:
            raise ValueError("Vector database path required for player type 'lpml_llm'")
        return LPMLEnhancedLLMPlayer(
            vectordb_path=vectordb_path,
            model_name=llm_model_name,
            name="LPML-Enhanced LLM",
        )
    else:
        raise ValueError(f"Unknown player type: {player_type}")


def save_results(results: Dict[str, Any], output_dir: str) -> None:
    """
    Save comparison results to files.
    
    Args:
        results: Dictionary containing comparison results
        output_dir: Directory to save results
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results to JSON file
    with open(os.path.join(output_dir, "comparison_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Create figures directory
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Set seaborn style
    sns.set(style="whitegrid")
    
    # Process each pair of opponents
    for pair_key, pair_results in results.items():
        player1 = pair_results["player1"]
        player2 = pair_results["player2"]
        
        # Plot win rates
        plt.figure(figsize=(10, 6))
        players = [player1["name"], player2["name"]]
        win_rates = [player1["win_rate"], player2["win_rate"]]
        
        ax = sns.barplot(x=players, y=win_rates, palette=["royalblue", "darkorange"])
        plt.xlabel("Player", fontsize=12)
        plt.ylabel("Win Rate", fontsize=12)
        plt.title(f"Win Rates: {player1['name']} vs {player2['name']}", fontsize=14)
        plt.ylim(0, 1)
        
        # Add value labels on top of bars
        for i, p in enumerate(ax.patches):
            ax.annotate(f"{p.get_height():.2f}", 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'bottom',
                       fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"{pair_key}_win_rates.png"))
        
        # Plot average scores
        plt.figure(figsize=(10, 6))
        avg_scores = [player1["avg_score"], player2["avg_score"]]
        
        ax = sns.barplot(x=players, y=avg_scores, palette=["royalblue", "darkorange"])
        plt.xlabel("Player", fontsize=12)
        plt.ylabel("Average Score", fontsize=12)
        plt.title(f"Average Scores: {player1['name']} vs {player2['name']}", fontsize=14)
        
        # Add value labels on top of bars
        for i, p in enumerate(ax.patches):
            ax.annotate(f"{p.get_height():.2f}", 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'bottom',
                       fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"{pair_key}_avg_scores.png"))
        
        # Plot average move counts
        plt.figure(figsize=(10, 6))
        avg_moves = [player1["avg_moves"], player2["avg_moves"]]
        
        ax = sns.barplot(x=players, y=avg_moves, palette=["royalblue", "darkorange"])
        plt.xlabel("Player", fontsize=12)
        plt.ylabel("Average Moves", fontsize=12)
        plt.title(f"Average Move Counts: {player1['name']} vs {player2['name']}", fontsize=14)
        
        # Add value labels on top of bars
        for i, p in enumerate(ax.patches):
            ax.annotate(f"{p.get_height():.2f}", 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'bottom',
                       fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"{pair_key}_avg_moves.png"))
        
        # Plot score distributions
        plt.figure(figsize=(12, 6))
        
        score_data = {
            player1["name"]: player1["scores"],
            player2["name"]: player2["scores"]
        }
        
        sns.histplot(data=score_data, bins=[-1.33, -0.67, 0, 0.67, 1.33], 
                    multiple="dodge", shrink=0.8, alpha=0.7)
        plt.xlabel("Score", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title(f"Score Distributions: {player1['name']} vs {player2['name']}", fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"{pair_key}_score_distributions.png"))
        
        # Plot move count distributions
        plt.figure(figsize=(12, 6))
        
        move_count_data = {
            player1["name"]: player1["move_counts"],
            player2["name"]: player2["move_counts"]
        }
        
        sns.histplot(data=move_count_data, bins=10, multiple="dodge", shrink=0.8, alpha=0.7)
        plt.xlabel("Move Count", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title(f"Move Count Distributions: {player1['name']} vs {player2['name']}", fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"{pair_key}_move_count_distributions.png"))
    
    # Create combined comparison plot
    plt.figure(figsize=(12, 8))
    
    # Extract win rates for each player
    player_names = []
    win_rates = []
    colors = []
    
    for pair_key, pair_results in results.items():
        player1 = pair_results["player1"]
        player2 = pair_results["player2"]
        
        player_names.extend([player1["name"], player2["name"]])
        win_rates.extend([player1["win_rate"], player2["win_rate"]])
        
        # Alternate colors for better visualization
        colors.extend(["royalblue", "darkorange"])
    
    # Create a DataFrame for seaborn
    import pandas as pd
    df = pd.DataFrame({
        "Player": player_names,
        "Win Rate": win_rates
    })
    
    # Plot all win rates in one chart
    ax = sns.barplot(x="Player", y="Win Rate", data=df, palette=colors)
    plt.title("Win Rates Comparison Across All Players", fontsize=16)
    plt.xlabel("Player", fontsize=14)
    plt.ylabel("Win Rate", fontsize=14)
    plt.ylim(0, 1)
    
    # Add value labels on top of bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{p.get_height():.2f}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'bottom',
                   fontsize=10)
    
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "combined_win_rates.png"))
    
    logger.info(f"Saved results to {output_dir}")


def create_report(results: Dict[str, Any], output_dir: str) -> None:
    """
    Create a report summarizing the comparison results.
    
    Args:
        results: Dictionary containing comparison results
        output_dir: Directory to save the report
    """
    # Create report header
    report = """# Connect Four Agent Comparison Report

## Summary

This report compares the performance of Normal LLM vs LPML-enhanced LLM.

The LPML-enhanced LLM uses retrieved language-based strategic knowledge as a reference to inspire its own decision-making. It doesn't blindly follow the strategies but uses them as additional context.

## Results

"""
    
    # Add results for each pair
    for pair_key, pair_results in results.items():
        player1 = pair_results["player1"]
        player2 = pair_results["player2"]
        
        # Calculate improvement if applicable
        if player1["type"] == "normal_llm" and player2["type"] == "lpml_llm":
            if player1["win_rate"] > 0:
                improvement = (player2["win_rate"] - player1["win_rate"]) / player1["win_rate"] * 100
            else:
                improvement = float("inf")
            improvement_text = f"(Improvement: {improvement:.2f}%)"
        else:
            improvement_text = ""
        
        report += f"### {player1['name']} vs {player2['name']}\n\n"
        
        # Add win rate comparison
        report += f"#### Win Rates {improvement_text}\n\n"
        report += f"![Win Rates](figures/{pair_key}_win_rates.png)\n\n"
        
        # Add table of results
        report += "| Metric | " + player1["name"] + " | " + player2["name"] + " |\n"
        report += "|--------|---------------|------------------|\n"
        report += f"| Win Rate | {player1['win_rate']:.2f} | {player2['win_rate']:.2f} |\n"
        report += f"| Average Score | {player1['avg_score']:.2f} | {player2['avg_score']:.2f} |\n"
        report += f"| Average Moves | {player1['avg_moves']:.2f} | {player2['avg_moves']:.2f} |\n\n"
        
        # Add game outcomes
        report += "#### Game Outcomes\n\n"
        report += "| Outcome | " + player1["name"] + " | " + player2["name"] + " |\n"
        report += "|---------|---------------|------------------|\n"
        report += f"| Wins | {player1['wins']} | {player2['wins']} |\n"
        report += f"| Losses | {player1['losses']} | {player2['losses']} |\n"
        report += f"| Draws | {player1['draws']} | {player2['draws']} |\n\n"
        
        # Add visualizations
        report += "#### Visualizations\n\n"
        report += f"![Average Scores](figures/{pair_key}_avg_scores.png)\n\n"
        report += f"![Average Move Counts](figures/{pair_key}_avg_moves.png)\n\n"
        report += f"![Score Distributions](figures/{pair_key}_score_distributions.png)\n\n"
        report += f"![Move Count Distributions](figures/{pair_key}_move_count_distributions.png)\n\n"
        
        # Add conclusion
        report += "#### Conclusion\n\n"
        if player1["win_rate"] > player2["win_rate"]:
            winner = player1["name"]
            loser = player2["name"]
        elif player2["win_rate"] > player1["win_rate"]:
            winner = player2["name"]
            loser = player1["name"]
        else:
            report += f"The {player1['name']} and {player2['name']} performed equally well, with identical win rates of {player1['win_rate']:.2f}.\n\n"
            continue
        
        margin = abs(player1["win_rate"] - player2["win_rate"])
        if margin > 0.2:
            strength = "significantly outperformed"
        elif margin > 0.1:
            strength = "outperformed"
        else:
            strength = "slightly outperformed"
        
        report += f"The {winner} {strength} the {loser}, with a win rate difference of {margin:.2f}.\n\n"
    
    # Add combined comparison
    report += "## Overall Comparison\n\n"
    report += "![Combined Win Rates](figures/combined_win_rates.png)\n\n"
    
    # Add final conclusion
    lpml_better = False
    for pair_key, pair_results in results.items():
        if "normal_llm" in pair_key and "lpml_llm" in pair_key:
            player1 = pair_results["player1"]
            player2 = pair_results["player2"]
            if player1["type"] == "lpml_llm" and player1["win_rate"] > player2["win_rate"]:
                lpml_better = True
            elif player2["type"] == "lpml_llm" and player2["win_rate"] > player1["win_rate"]:
                lpml_better = True
    
    report += "## Final Conclusion\n\n"
    
    if lpml_better:
        report += "The LPML-enhanced LLM agent demonstrated improved performance compared to the normal LLM agent. This suggests that retrieval of strategic knowledge via LPML annotations provides meaningful guidance for Connect Four gameplay decisions.\n\n"
        report += "This approach shows promise for enhancing LLM decision-making in strategic games and potentially in other domains where retrieving relevant domain knowledge can inform better decisions.\n"
    else:
        report += "The LPML-enhanced LLM agent did not show improved performance compared to the normal LLM agent in this evaluation. This suggests that the current approach to retrieving and utilizing strategic knowledge via LPML annotations may need refinement.\n\n"
        report += "Future work could explore improving the quality of LPML annotations, refining the retrieval mechanism, or adjusting how the retrieved knowledge is incorporated into the decision-making process.\n"
    
    # Save report
    with open(os.path.join(output_dir, "report.md"), "w") as f:
        f.write(report)
    
    logger.info(f"Created report in {output_dir}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Compare Connect Four agents with and without LPML-based RAG")
    
    parser.add_argument("--model", "-m", type=str, default=None,
                       help="Path to the trained PPO model (optional, only needed if using PPO agent)")
    parser.add_argument("--vectordb", "-v", type=str, default="data/vectordb",
                       help="Path to the vector database (default: data/vectordb)")
    parser.add_argument("--num-games", "-n", type=int, default=5,
                       help="Number of games to play for each agent pair (default: 5)")
    parser.add_argument("--opponent-pairs", type=str, default="normal_llm,lpml_llm;baby,normal_llm",
                       help="Semicolon-separated list of comma-separated player type pairs to compare (default: 'normal_llm,lpml_llm;baby,normal_llm')")
    parser.add_argument("--model-name", type=str, default="gpt-4o-mini",
                       help="Name of the LLM model to use for the LLM agents (default: gpt-4o-mini)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="OpenAI API key")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Directory to save results (default: results)")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed (default: 0)")
    
    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = parse_args()
    
    # The OpenAI API key will be automatically used from the environment variable
    # or when creating the OpenAI client in the player classes
    
    # Parse opponent pairs
    opponent_pairs = []
    for pair_str in args.opponent_pairs.split(";"):
        if "," in pair_str:
            player1, player2 = pair_str.split(",")
            opponent_pairs.append((player1.strip(), player2.strip()))
    
    if not opponent_pairs:
        opponent_pairs = [("normal_llm", "lpml_llm")]
    
    # Run comparison
    results = run_comparison(
        vectordb_path=args.vectordb,
        num_games=args.num_games,
        opponent_pairs=opponent_pairs,
        ppo_model_path=args.model,
        llm_model_name=args.model_name,
        seed=args.seed,
    )
    
    # Save results
    save_results(
        results=results,
        output_dir=args.results_dir,
    )
    
    # Create report
    create_report(
        results=results,
        output_dir=args.results_dir,
    )
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
