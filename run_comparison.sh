#!/bin/bash
# Run the full pipeline for training and comparing Connect Four agents
# with and without LPML-based RAG enhancement

# Setup directories
mkdir -p data/models data/trajectories data/lpml data/vectordb results

# Default settings (run all steps)
SKIP_DEPS=false
SKIP_TRAIN=false
SKIP_COLLECT=false
SKIP_LPML=false
SKIP_VECTORDB=false
SKIP_COMPARE=false
SKIP_PATTERNS=false

# Function to display help message
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help           Show this help message"
    echo "  --skip-deps          Skip installing dependencies"
    echo "  --skip-train         Skip training the PPO agent"
    echo "  --skip-collect       Skip collecting trajectories"
    echo "  --skip-lpml          Skip extracting LPML annotations"
    echo "  --skip-vectordb      Skip creating vector database"
    echo "  --skip-compare       Skip comparing agents"
    echo "  --skip-patterns      Skip extracting strategy patterns"
    echo "  --skip-all           Skip all steps (useful with selective enables)"
    echo "  --only-train         Run only the training step"
    echo "  --only-collect       Run only the trajectory collection step"
    echo "  --only-lpml          Run only the LPML extraction step"
    echo "  --only-vectordb      Run only the vector database creation step"
    echo "  --only-compare       Run only the agent comparison step"
    echo "  --only-patterns      Run only the strategy pattern extraction step"
    echo ""
    echo "Examples:"
    echo "  $0                   Run the full pipeline"
    echo "  $0 --skip-train      Run all steps except training"
    echo "  $0 --only-compare    Run only the comparison step"
    echo "  $0 --skip-all --only-lpml --only-vectordb    Run only LPML extraction and vector DB creation"
    exit 0
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        --skip-deps)
            SKIP_DEPS=true
            ;;
        --skip-train)
            SKIP_TRAIN=true
            ;;
        --skip-collect)
            SKIP_COLLECT=true
            ;;
        --skip-lpml)
            SKIP_LPML=true
            ;;
        --skip-vectordb)
            SKIP_VECTORDB=true
            ;;
        --skip-compare)
            SKIP_COMPARE=true
            ;;
        --skip-patterns)
            SKIP_PATTERNS=true
            ;;
        --skip-all)
            SKIP_DEPS=true
            SKIP_TRAIN=true
            SKIP_COLLECT=true
            SKIP_LPML=true
            SKIP_VECTORDB=true
            SKIP_COMPARE=true
            SKIP_PATTERNS=true
            ;;
        --only-train)
            SKIP_DEPS=true
            SKIP_COLLECT=true
            SKIP_LPML=true
            SKIP_VECTORDB=true
            SKIP_COMPARE=true
            SKIP_PATTERNS=true
            SKIP_TRAIN=false
            ;;
        --only-collect)
            SKIP_DEPS=true
            SKIP_TRAIN=true
            SKIP_LPML=true
            SKIP_VECTORDB=true
            SKIP_COMPARE=true
            SKIP_PATTERNS=true
            SKIP_COLLECT=false
            ;;
        --only-lpml)
            SKIP_DEPS=true
            SKIP_TRAIN=true
            SKIP_COLLECT=true
            SKIP_VECTORDB=true
            SKIP_COMPARE=true
            SKIP_PATTERNS=true
            SKIP_LPML=false
            ;;
        --only-vectordb)
            SKIP_DEPS=true
            SKIP_TRAIN=true
            SKIP_COLLECT=true
            SKIP_LPML=true
            SKIP_COMPARE=true
            SKIP_PATTERNS=true
            SKIP_VECTORDB=false
            ;;
        --only-compare)
            SKIP_DEPS=true
            SKIP_TRAIN=true
            SKIP_COLLECT=true
            SKIP_LPML=true
            SKIP_VECTORDB=true
            SKIP_PATTERNS=true
            SKIP_COMPARE=false
            ;;
        --only-patterns)
            SKIP_DEPS=true
            SKIP_TRAIN=true
            SKIP_COLLECT=true
            SKIP_LPML=true
            SKIP_VECTORDB=true
            SKIP_COMPARE=true
            SKIP_PATTERNS=false
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
    shift
done

# Check and install required dependencies
if [ "$SKIP_DEPS" = false ]; then
    echo "=== Checking and installing dependencies ==="
    # Install all required packages including the latest OpenAI SDK
    pip install gymnasium-connect-four stable-baselines3 "openai>=1.75" chromadb matplotlib tqdm seaborn pandas
    echo "=== Installed latest OpenAI SDK (>=1.75) required for o3 models ==="
fi

# Step 1: Train a PPO agent
if [ "$SKIP_TRAIN" = false ]; then
    echo "=== Step 1: Training PPO agent ==="
    python train.py \
        --timesteps 50000 \
        --opponent baby \
        --save-path data/models/ppo_connect4.zip
fi

# Step 2: Collect trajectories from the trained agent
if [ "$SKIP_COLLECT" = false ]; then
    echo "=== Step 2: Collecting trajectories ==="
    python collect_trajectories.py \
        --model data/models/ppo_connect4.zip \
        --episodes 10 \
        --output data/trajectories/trajectories.pkl
fi

# Step 3: Extract LPML annotations from the trajectories
if [ "$SKIP_LPML" = false ]; then
    echo "=== Step 3: Extracting LPML from trajectories ==="
    # Clear existing LPML file to avoid issues with incomplete data
    rm -f data/lpml/connect4_strategies.xml
    
    # Run with increased logging to see more details
    python -u extract_lpml.py \
        --input data/trajectories/trajectories.pkl \
        --output data/lpml/connect4_strategies.xml \
        --model o3-2025-04-16
fi

# Step 4: Create vector database from LPML annotations
if [ "$SKIP_VECTORDB" = false ]; then
    echo "=== Step 4: Creating vector database ==="
    python -m utils.xml_utils \
        --input data/lpml/connect4_strategies.xml \
        --output data/vectordb \
        --create-db
fi

# Step 5: Run comparison between agents
if [ "$SKIP_COMPARE" = false ]; then
    echo "=== Step 5: Running agent comparison ==="
    python compare_agents.py \
        --vectordb data/vectordb \
        --num-games 20 \
        --opponent-pairs "normal_llm,lpml_llm" \
        --model-name gpt-4o-mini \
        --results-dir results
fi

# Optional: Extract strategy patterns from LPML
if [ "$SKIP_PATTERNS" = false ]; then
    echo "=== Extracting strategy patterns ==="
    python -m utils.xml_utils \
        --input data/lpml/connect4_strategies.xml \
        --extract-patterns > results/strategy_patterns.json
fi

echo "=== Pipeline completed ==="
echo "Results saved to the 'results' directory"
