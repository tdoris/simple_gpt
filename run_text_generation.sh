#!/bin/bash

# Script to run text generation using a pre-trained model

# Default values
MODEL_PATH="./output/slimpajama_arxiv_20250324_084707"  # Using a better default model
PROMPT="Once upon a time"
MAX_LENGTH=100
TEMPERATURE=0.6  # Lower temperature for more coherent outputs
TOP_K=40  # Lower top_k for better focus
TOP_P=0.92  # Slightly lower top_p for more focused sampling
DO_SAMPLE="--do_sample"
NUM_SEQUENCES=1
REPETITION_PENALTY=1.5  # Increased repetition penalty to reduce repetitions
MIN_LENGTH=0

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top_k)
            TOP_K="$2"
            shift 2
            ;;
        --top_p)
            TOP_P="$2"
            shift 2
            ;;
        --greedy)
            DO_SAMPLE=""
            shift
            ;;
        --num_seqs)
            NUM_SEQUENCES="$2"
            shift 2
            ;;
        --rep_penalty)
            REPETITION_PENALTY="$2"
            shift 2
            ;;
        --min_length)
            MIN_LENGTH="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model <path>        Path to the model directory (default: ./output/gpt2_weights)"
            echo "  --prompt <text>       Prompt for text generation (default: 'Once upon a time')"
            echo "  --length <num>        Maximum length of generated text (default: 100)"
            echo "  --temperature <float> Temperature for sampling (default: 0.7)"
            echo "  --top_k <num>         Top-k filtering parameter (default: 50)"
            echo "  --top_p <float>       Top-p filtering parameter (default: 0.95)"
            echo "  --greedy              Use greedy decoding instead of sampling"
            echo "  --num_seqs <num>      Number of sequences to generate (default: 1)"
            echo "  --rep_penalty <float> Repetition penalty (default: 1.2, higher values reduce repetition)"
            echo "  --min_length <num>    Minimum length of generated text (default: 0)"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage information."
            exit 1
            ;;
    esac
done

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Model directory $MODEL_PATH does not exist."
    echo "Use fetch_gpt2_weights.py to download a pre-trained model or provide a valid model path."
    exit 1
fi

# Set environment variables for cache directories
export HF_HOME="./cache/huggingface"
export HF_DATASETS_CACHE="./cache/datasets"

# Build the command
GENERATE_CMD="python -m simple_gpt.scripts.generate \
  --model_path=\"$MODEL_PATH\" \
  --prompt=\"$PROMPT\" \
  --max_length=$MAX_LENGTH \
  --temperature=$TEMPERATURE \
  --top_k=$TOP_K \
  --top_p=$TOP_P \
  --num_return_sequences=$NUM_SEQUENCES \
  --repetition_penalty=$REPETITION_PENALTY \
  --min_length=$MIN_LENGTH \
  $DO_SAMPLE"

# Run the command
echo "Generating text with the following configuration:"
echo "- Model: $MODEL_PATH"
echo "- Prompt: \"$PROMPT\""
echo "- Max length: $MAX_LENGTH"
echo "- Temperature: $TEMPERATURE"
echo "- Top-k: $TOP_K"
echo "- Top-p: $TOP_P"
echo "- Sampling: ${DO_SAMPLE:+Enabled}"
echo "- Number of sequences: $NUM_SEQUENCES"
echo "- Repetition penalty: $REPETITION_PENALTY"
echo "- Minimum length: $MIN_LENGTH"
echo ""
echo "Running command: $GENERATE_CMD"
eval $GENERATE_CMD
