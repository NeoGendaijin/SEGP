#!/usr/bin/env python
"""
Train a PPO agent to play Connect Four.

This script implements PPO training for a Connect Four agent, with options for
different opponents, hyperparameters, and evaluation methods.
"""

import os
import time
import logging
import argparse
import numpy as np
from typing import Dict, Any, List, Optional

# Import stable-baselines3
from stable_baselines3 import PPO

# Import Connect Four environment
from connect_four_gymnasium import ConnectFourEnv
from connect_four_gymnasium.players import (
    BabyPlayer, BabySmarterPlayer, ChildPlayer, ChildSmarterPlayer,
    TeenagerPlayer, TeenagerSmarterPlayer, AdultPlayer, AdultSmarterPlayer,
    ModelPlayer
)

# Import ELO leaderboard for evaluation
from connect_four_gymnasium.tools import EloLeaderboard

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


def train_ppo(
    opponent_name: str = "baby",
    total_timesteps: int = 1000000,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    ent_coef: float = 0.01,
    seed: int = 0,
    device: str = "auto",
    verbose: int = 1,
    self_play: bool = False,
    save_path: Optional[str] = None,
):
    """
    Train a PPO agent for Connect Four.
    
    Args:
        opponent_name: Name of the opponent during training
        total_timesteps: Total timesteps for training
        learning_rate: Learning rate
        gamma: Discount factor
        ent_coef: Entropy coefficient for the loss calculation
        seed: Random seed
        device: Device to run the model on ('auto', 'cpu', 'cuda', 'cuda:0', etc.)
        verbose: Verbosity level
        self_play: Whether to use self-play
        save_path: Path to save the trained model
        
    Returns:
        Trained PPO model
    """
    # Set random seed
    np.random.seed(seed)
    
    # Create environment with specified opponent
    if self_play:
        # For self-play, we first create an environment without opponent
        env = ConnectFourEnv()
        
        # Create model
        model = PPO("MlpPolicy", env, 
                    learning_rate=learning_rate,
                    gamma=gamma,
                    ent_coef=ent_coef,
                    verbose=verbose,
                    seed=seed,
                    device=device)
        
        # Then set the opponent to be the model itself
        opponent = ModelPlayer(model, name="yourself")
        env.change_opponent(opponent)
    else:
        # Create opponent
        opponent = get_player_by_name(opponent_name)
        
        # Create environment
        env = ConnectFourEnv(opponent=opponent)
        
        # Create model
        model = PPO("MlpPolicy", env, 
                    learning_rate=learning_rate,
                    gamma=gamma,
                    ent_coef=ent_coef,
                    verbose=verbose,
                    seed=seed,
                    device=device)
    
    # Train model
    logger.info(f"Starting training for {total_timesteps} timesteps")
    start_time = time.time()
    
    model.learn(total_timesteps=total_timesteps)
    
    training_time = time.time() - start_time
    logger.info(f"Finished training in {training_time:.2f} seconds")
    
    # Save model if requested
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        model.save(save_path)
        logger.info(f"Saved model to {save_path}")
    
    return model


def evaluate_model(
    model: PPO,
    num_matches: int = 200,
):
    """
    Evaluate a trained model using ELO rating.
    
    Args:
        model: Trained PPO model
        num_matches: Number of matches for ELO evaluation
        
    Returns:
        ELO rating
    """
    logger.info(f"Evaluating model with {num_matches} matches")
    
    # Create a ModelPlayer from the model
    model_player = ModelPlayer(model, name="trained_model")
    
    # Get ELO rating
    elo = EloLeaderboard().get_elo(model_player, num_matches=num_matches)
    
    logger.info(f"Model ELO rating: {elo}")
    
    return elo


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train a PPO agent for Connect Four")
    
    # Training parameters
    parser.add_argument("--timesteps", "-t", type=int, default=1000000,
                       help="Total timesteps for training (default: 1,000,000)")
    parser.add_argument("--learning-rate", "-lr", type=float, default=3e-4,
                       help="Learning rate (default: 3e-4)")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor (default: 0.99)")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                       help="Entropy coefficient (default: 0.01)")
    
    # Opponent and environment parameters
    parser.add_argument("--opponent", "-o", type=str, default="baby",
                       choices=["baby", "babysmarter", "child", "childsmarter", 
                                "teenager", "teenagersmarter", "adult", "adultsmarter"],
                       help="Opponent during training (default: baby)")
    parser.add_argument("--self-play", action="store_true",
                       help="Use self-play training")
    
    # Output parameters
    parser.add_argument("--save-path", "-s", type=str, default="data/models/ppo_connect4.zip",
                       help="Path to save the trained model (default: data/models/ppo_connect4.zip)")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed (default: 0)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to run the model on (default: auto)")
    parser.add_argument("--verbose", type=int, default=1,
                       help="Verbosity level (default: 1)")
    parser.add_argument("--evaluate", action="store_true",
                       help="Evaluate the model after training")
    parser.add_argument("--num-matches", type=int, default=200,
                       help="Number of matches for evaluation (default: 200)")
    
    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = parse_args()
    
    # Train model
    model = train_ppo(
        opponent_name=args.opponent,
        total_timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        seed=args.seed,
        device=args.device,
        verbose=args.verbose,
        self_play=args.self_play,
        save_path=args.save_path,
    )
    
    # Evaluate model if requested
    if args.evaluate:
        elo = evaluate_model(
            model=model,
            num_matches=args.num_matches,
        )
        
        # Print evaluation results
        print("\n===== Evaluation Results =====")
        print(f"ELO rating: {elo}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
