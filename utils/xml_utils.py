#!/usr/bin/env python
"""
XML utilities for handling LPML (LLM-Prompting Markup Language) data.

This module provides functions for parsing, querying, and indexing LPML data
stored in XML format, particularly for Connect Four strategy extraction.
"""

import os
import sys
import json
import pickle
import argparse
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple

# Check if Chroma is available
try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


def parse_lpml_file(file_path: str) -> ET.Element:
    """
    Parse an LPML XML file.
    
    Args:
        file_path: Path to the LPML XML file
        
    Returns:
        Root element of the parsed XML
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"LPML file not found: {file_path}")
    
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        return root
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML in LPML file: {e}")


def extract_turn_data(turn_element: ET.Element) -> Dict[str, Any]:
    """
    Extract data from a Turn element in the LPML.
    
    Args:
        turn_element: XML element representing a turn
        
    Returns:
        Dictionary containing the turn data
    """
    turn_data = {}
    
    # Get turn number
    turn_number = turn_element.get("number")
    if turn_number:
        turn_data["turn_number"] = int(turn_number)
    
    # Extract Condition, Thought, Execution, and Action elements
    for element_name in ["Condition", "Thought", "Execution", "Action"]:
        element = turn_element.find(element_name)
        if element is not None and element.text:
            turn_data[element_name.lower()] = element.text.strip()
    
    return turn_data


def extract_all_turns(root: ET.Element) -> List[Dict[str, Any]]:
    """
    Extract data from all Turn elements in the LPML.
    
    Args:
        root: Root element of the parsed XML
        
    Returns:
        List of dictionaries containing turn data
    """
    turns = []
    
    # Handle single LPML document
    if root.tag == "LPML":
        for turn_element in root.findall("Turn"):
            turn_data = extract_turn_data(turn_element)
            turn_data["trajectory_id"] = root.get("trajectory_id", "0")
            turns.append(turn_data)
    
    # Handle collection of LPML documents
    elif root.tag == "LPML_Collection":
        for lpml_element in root.findall("LPML"):
            trajectory_id = lpml_element.get("trajectory_id", "0")
            for turn_element in lpml_element.findall("Turn"):
                turn_data = extract_turn_data(turn_element)
                turn_data["trajectory_id"] = trajectory_id
                turns.append(turn_data)
    
    return turns


def extract_strategies(lpml_file: str) -> List[Dict[str, Any]]:
    """
    Extract all strategies from an LPML file.
    
    Args:
        lpml_file: Path to the LPML XML file
        
    Returns:
        List of dictionaries containing strategy data
    """
    root = parse_lpml_file(lpml_file)
    return extract_all_turns(root)


def create_condition_index(strategies: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Create an index of strategies by condition.
    
    Args:
        strategies: List of strategy dictionaries
        
    Returns:
        Dictionary mapping conditions to strategies
    """
    condition_index = {}
    
    for strategy in strategies:
        if "condition" in strategy:
            condition = strategy["condition"]
            if condition not in condition_index:
                condition_index[condition] = []
            condition_index[condition].append(strategy)
    
    return condition_index


def find_similar_conditions(query_condition: str, strategies: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Find strategies with similar conditions (simple text overlap approach).
    
    Note: This is a simple implementation. For production, use embedding similarity.
    
    Args:
        query_condition: Condition text to match
        strategies: List of strategy dictionaries
        top_k: Number of top matches to return
        
    Returns:
        List of matching strategy dictionaries
    """
    scores = []
    
    for strategy in strategies:
        if "condition" in strategy:
            condition = strategy["condition"]
            
            # Simple word overlap score
            query_words = set(query_condition.lower().split())
            condition_words = set(condition.lower().split())
            overlap = len(query_words.intersection(condition_words))
            
            scores.append((overlap, strategy))
    
    # Sort by score in descending order
    scores.sort(reverse=True, key=lambda x: x[0])
    
    # Return top k matches
    return [strategy for _, strategy in scores[:top_k]]


def strategies_to_xml(strategies: List[Dict[str, Any]], output_file: str) -> None:
    """
    Convert a list of strategies to XML and save to a file.
    
    Args:
        strategies: List of strategy dictionaries
        output_file: Path to save the XML file
    """
    # Create root element
    root = ET.Element("LPML_Collection")
    
    # Group strategies by trajectory_id
    trajectories = {}
    for strategy in strategies:
        trajectory_id = strategy.get("trajectory_id", "0")
        if trajectory_id not in trajectories:
            trajectories[trajectory_id] = []
        trajectories[trajectory_id].append(strategy)
    
    # Create XML structure
    for trajectory_id, trajectory_strategies in trajectories.items():
        lpml = ET.SubElement(root, "LPML", {"trajectory_id": str(trajectory_id)})
        
        for strategy in trajectory_strategies:
            # Get turn number if available, otherwise use index
            turn_number = str(strategy.get("turn_number", 0))
            turn = ET.SubElement(lpml, "Turn", {"number": turn_number})
            
            for key in ["condition", "thought", "execution", "action"]:
                if key in strategy:
                    elem = ET.SubElement(turn, key.capitalize())
                    elem.text = strategy[key]
    
    # Create XML tree and save to file
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)


def create_vector_db(xml_file: str, output_dir: str) -> None:
    """
    Create a vector database from an LPML file.
    
    Args:
        xml_file: Path to the LPML XML file
        output_dir: Directory to save the vector database
    """
    if not CHROMA_AVAILABLE:
        print("Error: chromadb not installed. Install with 'pip install chromadb'")
        return
        
    # Extract strategies from XML
    strategies = extract_strategies(xml_file)
    print(f"Extracted {len(strategies)} strategies from {xml_file}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create embedding function
    embedding_func = embedding_functions.DefaultEmbeddingFunction()
    
    # Create persistent client
    chroma_path = os.path.join(output_dir, "chroma")
    client = chromadb.PersistentClient(path=chroma_path)
    
    # Check if collection already exists and delete it
    try:
        existing_collections = client.list_collections()
        for collection in existing_collections:
            if collection.name == "lpml_strategies":
                print(f"Collection 'lpml_strategies' already exists, deleting it first...")
                client.delete_collection(name="lpml_strategies")
                break
    except Exception as e:
        print(f"Warning when checking existing collections: {e}")
    
    # Create collection
    collection = client.create_collection(
        name="lpml_strategies",
        embedding_function=embedding_func,
        metadata={"description": "Connect Four LPML strategies"}
    )
    
    # Add documents to collection
    documents = []
    metadatas = []
    ids = []
    
    for i, strategy in enumerate(strategies):
        if "condition" not in strategy or not strategy["condition"]:
            continue
            
        # Combine condition and thought for context
        document = f"Board state: {strategy['condition']}\n"
        
        if "thought" in strategy and strategy["thought"]:
            document += f"Strategic analysis: {strategy['thought']}\n"
            
        if "execution" in strategy and strategy["execution"]:
            document += f"Move execution: {strategy['execution']}\n"
            
        if "action" in strategy and strategy["action"]:
            document += f"Action: {strategy['action']}"
            
        # Add metadata
        metadata = {
            "turn_number": strategy.get("turn_number", 0),
            "trajectory_id": strategy.get("trajectory_id", "0")
        }
        
        doc_id = f"strategy_{i}"
        
        documents.append(document)
        metadatas.append(metadata)
        ids.append(doc_id)
    
    # Add documents in batches
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch_end = min(i + batch_size, len(documents))
        collection.add(
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end],
            ids=ids[i:batch_end]
        )
        
    print(f"Created vector database with {len(documents)} documents")
    print(f"Vector database saved to {chroma_path}")
    
    # Save strategies to a pickle file for future reference
    strategies_path = os.path.join(output_dir, "strategies.pkl")
    with open(strategies_path, "wb") as f:
        pickle.dump(strategies, f)
    
    print(f"Strategies saved to {strategies_path}")


def search_vector_db(db_path: str):
    """
    Load a vector database for searching.
    
    Args:
        db_path: Path to the vector database directory
        
    Returns:
        The loaded vector database
    """
    if not CHROMA_AVAILABLE:
        print("Error: chromadb not installed. Install with 'pip install chromadb'")
        return None
        
    # Check if the directory exists
    chroma_path = os.path.join(db_path, "chroma")
    if not os.path.exists(chroma_path):
        print(f"Error: Vector database not found at {chroma_path}")
        return None
        
    # Create embedding function
    embedding_func = embedding_functions.DefaultEmbeddingFunction()
    
    # Create persistent client
    client = chromadb.PersistentClient(path=chroma_path)
    
    # Get collection
    try:
        collection = client.get_collection(
            name="lpml_strategies",
            embedding_function=embedding_func
        )
        print(f"Loaded vector database from {chroma_path}")
        return collection
    except Exception as e:
        print(f"Error loading vector database: {e}")
        return None


def extract_strategy_patterns(xml_file: str) -> Dict[str, Any]:
    """
    Extract strategy patterns from an LPML file.
    
    Args:
        xml_file: Path to the LPML XML file
        
    Returns:
        Dictionary containing strategy patterns
    """
    # Extract strategies from XML
    strategies = extract_strategies(xml_file)
    
    # Extract common patterns
    patterns = {
        "opening_moves": [],
        "threats": [],
        "blocks": [],
        "setups": [],
        "winning_moves": []
    }
    
    for strategy in strategies:
        thought = strategy.get("thought", "")
        
        # Check for opening moves (typically in early turns)
        if strategy.get("turn_number", 0) <= 2 and thought:
            patterns["opening_moves"].append({
                "turn": strategy.get("turn_number", 0),
                "thought": thought,
                "action": strategy.get("action", "")
            })
        
        # Check for threats and blocks
        if "threat" in thought.lower():
            patterns["threats"].append({
                "turn": strategy.get("turn_number", 0),
                "thought": thought,
                "action": strategy.get("action", "")
            })
        
        if "block" in thought.lower():
            patterns["blocks"].append({
                "turn": strategy.get("turn_number", 0),
                "thought": thought,
                "action": strategy.get("action", "")
            })
            
        # Check for setups
        if "setup" in thought.lower() or "setting up" in thought.lower():
            patterns["setups"].append({
                "turn": strategy.get("turn_number", 0),
                "thought": thought,
                "action": strategy.get("action", "")
            })
            
        # Check for winning moves
        if "win" in thought.lower() or "winning" in thought.lower():
            patterns["winning_moves"].append({
                "turn": strategy.get("turn_number", 0),
                "thought": thought,
                "action": strategy.get("action", "")
            })
    
    # Calculate statistics
    total_turns = len(strategies)
    if total_turns > 0:
        patterns["stats"] = {
            "total_turns": total_turns,
            "opening_move_percentage": len(patterns["opening_moves"]) / total_turns * 100,
            "threat_percentage": len(patterns["threats"]) / total_turns * 100,
            "block_percentage": len(patterns["blocks"]) / total_turns * 100,
            "setup_percentage": len(patterns["setups"]) / total_turns * 100,
            "winning_move_percentage": len(patterns["winning_moves"]) / total_turns * 100
        }
    
    return patterns


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="XML utilities for LPML data")
    
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Path to the input LPML XML file")
    
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Path for the output file or directory")
    
    parser.add_argument("--create-db", action="store_true",
                       help="Create a vector database from the LPML file")
    
    parser.add_argument("--extract-patterns", action="store_true",
                       help="Extract strategy patterns from the LPML file")
    
    args = parser.parse_args()
    
    # Check input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    # Create vector database
    if args.create_db:
        if not args.output:
            print("Error: Output directory required for --create-db")
            return 1
            
        create_vector_db(args.input, args.output)
        return 0
    
    # Extract strategy patterns
    if args.extract_patterns:
        patterns = extract_strategy_patterns(args.input)
        print(json.dumps(patterns, indent=2))
        return 0
    
    # Default: print extracted strategies
    strategies = extract_strategies(args.input)
    print(f"Extracted {len(strategies)} strategies from {args.input}")
    
    # Print first strategy
    if strategies:
        first = strategies[0]
        print("\nFirst strategy:")
        for key, value in first.items():
            if key != "condition" and key != "thought":  # Skip long text
                print(f"{key}: {value}")
        
        # Print condition preview
        if "condition" in first:
            print(f"condition: {first['condition'][:100]}...")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
