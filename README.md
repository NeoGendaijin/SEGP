# SEGP: Strategic Extraction and Generation Pipeline

This project implements a pipeline for enhancing reinforcement learning agents with language-based strategic knowledge using LPML (LLM-Prompting Markup Language) annotations and a Retrieval Augmented Generation (RAG) approach.

## Overview

The pipeline consists of the following components:

1. **Training**: Train a PPO agent to play Connect Four.
2. **Trajectory Collection**: Collect game trajectories from the trained agent.
3. **LPML Extraction**: Use a language model to generate LPML annotations that describe the strategic reasoning behind each move.
4. **Vector Database Creation**: Create a vector database from the LPML annotations for efficient retrieval.
5. **Agent Comparison**: Compare the performance of a vanilla agent against an agent enhanced with LPML-based RAG.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required packages: `gymnasium-connect-four`, `stable-baselines3`, `openai`, `chromadb`, `matplotlib`, `tqdm`

### Installation

```bash
pip install gymnasium-connect-four stable-baselines3 openai chromadb matplotlib tqdm
```

### Running the Pipeline

To run the complete pipeline, execute:

```bash
bash run_comparison.sh
```

This script will:
1. Train a PPO agent for Connect Four
2. Collect trajectories from the trained agent
3. Extract LPML annotations from the trajectories
4. Create a vector database from the LPML annotations
5. Compare the vanilla agent with the RAG-enhanced agent
6. Generate a report with the results

### Skip Options for run_comparison.sh

You can skip specific steps of the pipeline using command-line options:

```bash
# Skip training step (use existing model)
bash run_comparison.sh --skip-train

# Run only the comparison step
bash run_comparison.sh --only-compare

# Run only LPML extraction and vector database creation
bash run_comparison.sh --skip-all --only-lpml --only-vectordb
```

Available options:
- `--skip-deps`: Skip installing dependencies
- `--skip-train`: Skip training the PPO agent
- `--skip-collect`: Skip collecting trajectories
- `--skip-lpml`: Skip extracting LPML annotations
- `--skip-vectordb`: Skip creating vector database
- `--skip-compare`: Skip comparing agents
- `--skip-patterns`: Skip extracting strategy patterns
- `--skip-all`: Skip all steps (useful with selective enables)
- `--only-train`: Run only the training step
- `--only-collect`: Run only the trajectory collection step
- `--only-lpml`: Run only the LPML extraction step
- `--only-vectordb`: Run only the vector database creation step
- `--only-compare`: Run only the agent comparison step
- `--only-patterns`: Run only the strategy pattern extraction step

Run `bash run_comparison.sh --help` for more information.

## Components in Detail

### 1. Training (train.py)

Trains a PPO agent for Connect Four against different opponents:

```bash
python train.py --timesteps 50000 --opponent baby --save-path data/models/ppo_connect4.zip
```

Options:
- `--timesteps`: Number of timesteps for training (default: 1,000,000)
- `--opponent`: Opponent to train against (default: baby)
- `--self-play`: Use self-play training
- `--save-path`: Path to save the trained model

### 2. Trajectory Collection (collect_trajectories.py)

Collects game trajectories from the trained agent:

```bash
python collect_trajectories.py --model data/models/ppo_connect4.zip --episodes 10 --output data/trajectories/trajectories.pkl
```

Options:
- `--model`: Path to the trained model
- `--episodes`: Number of episodes to collect (default: 50)
- `--opponent`: Opponent to play against (default: baby)
- `--output`: Path to save the trajectories

### 3. LPML Extraction (extract_lpml.py)

Generates LPML annotations from the collected trajectories:

```bash
python extract_lpml.py --input data/trajectories/trajectories.pkl --output data/lpml/connect4_strategies.xml --model gpt-4o-mini
```

Options:
- `--input`: Path to the trajectories file
- `--output`: Path to save the LPML file
- `--model`: Model to use for LPML generation (default: gpt-4o-mini)
- `--api-key`: OpenAI API key

### 4. Vector Database Creation (utils/xml_utils.py)

Creates a vector database from the LPML annotations:

```bash
python -m utils.xml_utils --input data/lpml/connect4_strategies.xml --output data/vectordb --create-db
```

Options:
- `--input`: Path to the LPML file
- `--output`: Path to save the vector database
- `--create-db`: Create a vector database

### 5. Agent Comparison (compare_agents.py)

Compares the vanilla agent with the RAG-enhanced agent:

```bash
python compare_agents.py --model data/models/ppo_connect4.zip --vectordb data/vectordb --num-games 5 --opponent baby --model-name gpt-4o-mini --results-dir results
```

Options:
- `--model`: Path to the trained model
- `--vectordb`: Path to the vector database
- `--num-games`: Number of games to play for each player (default: 20)
- `--opponent`: Opponent to play against (default: baby)
- `--model-name`: Name of the LLM model to use for RAG (default: gpt-4o-mini)
- `--results-dir`: Directory to save results (default: results)

## LPML Format

LPML (LLM-Prompting Markup Language) is an XML format used to annotate strategic reasoning in games. For Connect Four, each turn is annotated with:

- **Condition**: Description of the board state
- **Thought**: Strategic analysis of possible moves
- **Execution**: How the chosen move should be executed
- **Action**: The chosen action (column)

Example:
```xml
<LPML trajectory_id="0">
  <Turn number="1">
    <Condition>Empty board, player's turn to move.</Condition>
    <Thought>The center column is strategically important as it provides the most opportunities for connecting four.</Thought>
    <Execution>Place a piece in column 3.</Execution>
    <Action>3</Action>
  </Turn>
</LPML>
```

## Results

After running the comparison, results are saved in the `results` directory:
- `comparison_results.json`: Raw comparison results
- `report.md`: Summary report with visualizations
- `figures/`: Visualizations of the results
- `strategy_patterns.json`: Extracted strategy patterns from LPML

## Citation

If you use this code in your research, please cite:

```
@misc{segp,
  author = {SEGP Team},
  title = {Strategic Extraction and Generation Pipeline},
  year = {2023},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/username/segp}}
}
```
