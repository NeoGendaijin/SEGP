#!/usr/bin/env python
"""
Extract LPML annotations from Connect Four game trajectories.

This script loads trajectories collected from Connect Four games and uses
a language model to generate LPML (LLM-Prompting Markup Language) annotations
describing the strategy and reasoning behind each move.
"""

import os
import json
import pickle
import logging
import argparse
import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import tqdm

# Import OpenAI for LPML generation
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

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


def generate_lpml_with_openai(
    trajectory: List[Dict[str, Any]],
    model_name: str = "gpt-4o-mini",
    max_retries: int = 3,
    retry_delay: float = 1.0,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate LPML for a trajectory using OpenAI.
    
    Args:
        trajectory: List of steps in the trajectory
        model_name: OpenAI model to use for generation
        max_retries: Maximum number of retries on failure
        retry_delay: Delay between retries in seconds
        
    Returns:
        Dictionary containing LPML annotations for each turn
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI package not available. Install with 'pip install openai'")
    
    # Build system prompt
    system_prompt = """
    You are an expert Connect Four strategy annotator. You will analyze game states and explain the strategic reasoning for each move.
    For each turn, you'll provide:
    1. A concise description of the current board state (Condition)
    2. Strategic analysis of the best move options (Thought)
    3. How that move should be executed (Execution)
    
    Keep explanations clear and focused on Connect Four strategy concepts.
    """
    
    # Build user prompt with the trajectory data
    user_prompt = "Analyze this Connect Four game trajectory:\n\n"
    
    for step in trajectory:
        board_str = board_to_string(step["obs"])
        action = step["action"]
        user_prompt += f"Turn {step['turn'] + 1}:\n{board_str}\nAction: {action}\n\n"
    
    # Request format instructions
    user_prompt += """
    For each turn, provide your analysis in this format:
    
    Turn [number]:
    Condition: [Description of the board state]
    Thought: [Strategic analysis of possible moves]
    Execution: [How the chosen move should be executed]
    """
    
    # Call OpenAI API with retries
    for attempt in range(max_retries):
        try:
            # Create a standard client with API key if provided
            client = OpenAI(api_key=api_key) if api_key else OpenAI()
            
            # Use different parameters depending on the model
            if model_name.startswith("o3"):
                # For o3 Reasoning models
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "developer", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_completion_tokens=4000,
                    reasoning_effort="high",
                )
            else:
                # For traditional models like gpt-4o-mini
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=4000,
                )
            
            # Extract response text
            lpml_text = response.choices[0].message.content
            
            # Debug: Log the full response for the first trajectory
            if trajectory[0]["turn"] == 0:
                logger.info(f"Raw o3 response:\n{lpml_text}")
            
            # Parse the response into LPML structure
            turns = {}
            current_turn = None
            current_section = None
            
            # First check if we have any valid content
            if not lpml_text or "Turn" not in lpml_text:
                # Manual fallback for empty or invalid responses
                for i, step in enumerate(trajectory):
                    turn_num = i + 1
                    turns[turn_num] = {
                        "condition": f"Board state at turn {turn_num}",
                        "thought": f"Strategic analysis for move to column {step['action']}",
                        "execution": f"Place piece in column {step['action']}",
                        "action": step['action']
                    }
                logger.warning(f"Using fallback LPML for trajectory starting at turn {trajectory[0]['turn']}")
                return turns
            
            # Regular parsing
            for line in lpml_text.strip().split('\n'):
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Check for turn header
                if line.startswith("Turn "):
                    try:
                        # Handle various formats: "Turn 1:", "Turn 1", etc.
                        parts = line.split()
                        if len(parts) >= 2:
                            # Extract number and remove any trailing colon
                            turn_str = parts[1].rstrip(':')
                            turn_num = int(turn_str)
                            current_turn = turn_num
                            turns[current_turn] = {
                                "condition": "",
                                "thought": "",
                                "execution": "",
                                "action": trajectory[turn_num - 1]["action"] if turn_num <= len(trajectory) else 0
                            }
                            logger.debug(f"Found Turn {turn_num}")
                    except (IndexError, ValueError) as e:
                        logger.warning(f"Error parsing turn header '{line}': {e}")
                        continue
                
                # Check for section headers
                elif current_turn is not None and line.startswith("Condition:"):
                    current_section = "condition"
                    turns[current_turn][current_section] = line[len("Condition:"):].strip()
                elif current_turn is not None and line.startswith("Thought:"):
                    current_section = "thought"
                    turns[current_turn][current_section] = line[len("Thought:"):].strip()
                elif current_turn is not None and line.startswith("Execution:"):
                    current_section = "execution"
                    turns[current_turn][current_section] = line[len("Execution:"):].strip()
                
                # If we're in a section and it's continued text, append it
                elif current_turn is not None and current_section is not None:
                    turns[current_turn][current_section] += " " + line
            
            # Fallback if parsing didn't produce any turns
            if not turns:
                logger.warning("Parsing failed to extract any turns, using fallback")
                for i, step in enumerate(trajectory):
                    turn_num = i + 1
                    turns[turn_num] = {
                        "condition": f"Board state at turn {turn_num}",
                        "thought": f"Strategic analysis for move to column {step['action']}",
                        "execution": f"Place piece in column {step['action']}",
                        "action": step['action']
                    }
            
            return turns
            
        except Exception as e:
            logger.warning(f"Error calling OpenAI API (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise
    
    # Should never reach here due to the raise in the loop
    return {}


def create_lpml_xml(
    trajectories: List[List[Dict[str, Any]]],
    lpml_annotations: List[Dict[str, Any]],
) -> ET.ElementTree:
    """
    Create an XML tree with LPML annotations.
    
    Args:
        trajectories: List of trajectories
        lpml_annotations: List of LPML annotations for each trajectory
        
    Returns:
        XML ElementTree with LPML annotations
    """
    # Create root element
    root = ET.Element("LPML_Collection")
    
    # Add trajectories
    for traj_idx, (trajectory, annotations) in enumerate(zip(trajectories, lpml_annotations)):
        # Create LPML element for this trajectory
        lpml = ET.SubElement(root, "LPML", {"trajectory_id": str(traj_idx)})
        
        # Add turns
        for turn_num, turn_data in annotations.items():
            # Create Turn element
            turn = ET.SubElement(lpml, "Turn", {"number": str(turn_num)})
            
            # Add Condition, Thought, Execution, and Action elements
            for key in ["condition", "thought", "execution", "action"]:
                if key in turn_data:
                    elem = ET.SubElement(turn, key.capitalize())
                    elem.text = str(turn_data[key])
    
    # Create XML tree
    tree = ET.ElementTree(root)
    
    return tree


def load_trajectories(input_path: str) -> List[List[Dict[str, Any]]]:
    """
    Load trajectories from a file.
    
    Args:
        input_path: Path to the trajectories file
        
    Returns:
        List of trajectories
    """
    with open(input_path, "rb") as f:
        trajectories = pickle.load(f)
    
    logger.info(f"Loaded {len(trajectories)} trajectories from {input_path}")
    
    return trajectories


def extract_lpml(
    trajectories: List[List[Dict[str, Any]]],
    model_name: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Extract LPML annotations from trajectories.
    
    Args:
        trajectories: List of trajectories
        model_name: Model to use for LPML generation
        
    Returns:
        List of LPML annotations for each trajectory
    """
    logger.info(f"Extracting LPML annotations for {len(trajectories)} trajectories using {model_name}")
    
    # Check if OpenAI is available
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI package not available. Install with 'pip install openai'")
    
    logger.info("Using OpenAI for LPML generation")
    
    # Extract LPML for each trajectory
    lpml_annotations = []
    
    for traj_idx, trajectory in enumerate(tqdm(trajectories, desc="Extracting LPML")):
        try:
            annotations = generate_lpml_with_openai(trajectory, model_name, api_key=api_key)
            lpml_annotations.append(annotations)
            
            # Log progress
            if (traj_idx + 1) % 10 == 0 or traj_idx == 0:
                logger.info(f"Extracted LPML for {traj_idx + 1}/{len(trajectories)} trajectories")
        
        except Exception as e:
            logger.error(f"Error extracting LPML for trajectory {traj_idx}: {e}")
            # Add empty annotations to maintain order
            lpml_annotations.append({})
    
    return lpml_annotations


def save_lpml(lpml_tree: ET.ElementTree, output_path: str) -> None:
    """
    Save LPML annotations to a file.
    
    Args:
        lpml_tree: XML ElementTree with LPML annotations
        output_path: Path to save the LPML XML file
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save XML
    lpml_tree.write(output_path, encoding="utf-8", xml_declaration=True)
    
    logger.info(f"Saved LPML annotations to {output_path}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Extract LPML annotations from Connect Four trajectories")
    
    parser.add_argument("--input", "-i", "--trajectories", type=str, required=True,
                       help="Path to the trajectories file")
    parser.add_argument("--output", "-o", type=str, default="data/lpml/connect4_strategies.xml",
                       help="Path to save the LPML file (default: data/lpml/connect4_strategies.xml)")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o-mini",
                       help="Model to use for LPML generation (default: gpt-4o-mini)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="OpenAI API key")
    
    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = parse_args()
    
    # No need to set api_key directly with the new OpenAI client pattern
    # The OpenAI client will automatically use OPENAI_API_KEY environment variable
    # or the key can be passed when creating the client
    
    # Load trajectories
    trajectories = load_trajectories(args.input)
    
    # Extract LPML annotations
    lpml_annotations = extract_lpml(
        trajectories=trajectories,
        model_name=args.model,
        api_key=args.api_key,
    )
    
    # Create LPML XML
    lpml_tree = create_lpml_xml(
        trajectories=trajectories,
        lpml_annotations=lpml_annotations,
    )
    
    # Save LPML
    save_lpml(
        lpml_tree=lpml_tree,
        output_path=args.output,
    )
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
