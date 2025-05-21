#!/usr/bin/env python
"""
Collect trajectories from a trained Connect Four agent.

This script loads a trained agent and collects game trajectories by having
the agent play against various opponents. These trajectories can then be
used for analysis or generating LPML annotations.
"""

import os
import pickle
import logging
import argparse
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Import stable-baselines3
from stable_baselines3 import PPO

# Import Connect Four environment
from connect_four_gymnasium import ConnectFourEnv
from connect_four_gymnasium.players import (
    BabyPlayer, BabySmarterPlayer, ChildPlayer, ChildSmarterPlayer,
    TeenagerPlayer, TeenagerSmarterPlayer, AdultPlayer, AdultSmarterPlayer,
    ModelPlayer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_player_by_name(player_name: str, model=None):
    """
    Get a player instance by name.
    
    Args:
        player_name: Name of the player type
        model: Model to use for ModelPlayer
        
    Returns:
        Player instance
        
    Raises:
        ValueError: If player_name is not recognized
    """
    player_name = player_name.lower()
    
    if player_name == "baby":
        return BabyPlayer()
    elif player_name == "babysmarter":
        return BabySmarterPlayer()
    elif player_name == "child":
        return ChildPlayer()
    elif player_name == "childsmarter":
        return ChildSmarterPlayer()
    elif player_name == "teenager":
        return TeenagerPlayer()
    elif player_name == "teenagersmarter":
        return TeenagerSmarterPlayer()
    elif player_name == "adult":
        return AdultPlayer()
    elif player_name == "adultsmarter":
        return AdultSmarterPlayer()
    elif player_name == "self" and model is not None:
        return ModelPlayer(model, name="self")
    elif player_name == "model" and model is not None:
        return ModelPlayer(model)
    else:
        raise ValueError(f"Unknown player type: {player_name}")


def collect_trajectories(
    model_path: str,
    num_episodes: int = 50,
    opponent_name: str = "baby",
    seed: int = 0,
    deterministic: bool = True,
) -> List[List[Dict[str, Any]]]:
    """
    Collect trajectories from a trained agent.
    
    Args:
        model_path: Path to the trained model
        num_episodes: Number of episodes to collect
        opponent_name: Name of the opponent to play against
        seed: Random seed
        deterministic: Whether to make deterministic decisions
        
    Returns:
        List of trajectories, where each trajectory is a list of steps
    """
    # Set random seed
    np.random.seed(seed)
    
    # Load model
    model = PPO.load(model_path)
    logger.info(f"Loaded model from {model_path}")
    
    # Create opponent
    opponent = get_player_by_name(opponent_name)
    logger.info(f"Using opponent: {opponent_name}")
    
    # Create environment
    env = ConnectFourEnv(opponent=opponent)
    
    # Initialize trajectories
    trajectories = []
    
    # Collect episodes
    for episode in tqdm(range(num_episodes), desc="Collecting episodes"):
        # Reset environment
        obs, info = env.reset(seed=seed + episode)
        
        # Initialize trajectory
        trajectory = []
        done = False
        
        # Play episode
        turn = 0
        while not done:
            # Get action from model
            action, _states = model.predict(obs, deterministic=deterministic)
            
            # Record pre-step state
            step = {
                "turn": turn,
                "obs": np.copy(obs),
                "action": int(action),
                "reward": 0.0,  # Will be updated after step
                "done": False,  # Will be updated after step
            }
            
            # Take step in environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Update step with results
            step["reward"] = float(reward)
            step["done"] = terminated or truncated
            
            # Add step to trajectory
            trajectory.append(step)
            
            # Update for next iteration
            obs = next_obs
            done = terminated or truncated
            turn += 1
        
        # Add trajectory to collection
        trajectories.append(trajectory)
        
        # Log progress
        if (episode + 1) % 10 == 0 or episode == 0:
            logger.info(f"Collected {episode + 1} episodes, latest has {len(trajectory)} turns")
    
    logger.info(f"Collected {len(trajectories)} trajectories with a total of {sum(len(t) for t in trajectories)} turns")
    
    return trajectories


def save_trajectories(trajectories: List[List[Dict[str, Any]]], output_path: str) -> None:
    """
    Save trajectories to a file.
    
    Args:
        trajectories: List of trajectories to save
        output_path: Path to save the trajectories
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save trajectories
    with open(output_path, "wb") as f:
        pickle.dump(trajectories, f)
    
    logger.info(f"Saved trajectories to {output_path}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Collect trajectories from a trained Connect Four agent")
    
    parser.add_argument("--model", "-m", type=str, required=True,
                       help="Path to the trained model")
    parser.add_argument("--episodes", "-e", type=int, default=50,
                       help="Number of episodes to collect (default: 50)")
    parser.add_argument("--opponent", "-o", type=str, default="baby",
                       choices=["baby", "babysmarter", "child", "childsmarter", 
                                "teenager", "teenagersmarter", "adult", "adultsmarter"],
                       help="Opponent to play against (default: baby)")
    parser.add_argument("--output", type=str, default="data/trajectories/trajectories.pkl",
                       help="Path to save the trajectories (default: data/trajectories/trajectories.pkl)")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed (default: 0)")
    parser.add_argument("--deterministic", action="store_true",
                       help="Use deterministic actions")
    
    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = parse_args()
    
    # Collect trajectories
    trajectories = collect_trajectories(
        model_path=args.model,
        num_episodes=args.episodes,
        opponent_name=args.opponent,
        seed=args.seed,
        deterministic=args.deterministic,
    )
    
    # Save trajectories
    save_trajectories(
        trajectories=trajectories,
        output_path=args.output,
    )
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
