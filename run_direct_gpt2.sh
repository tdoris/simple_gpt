#!/bin/bash

# Script to run text generation using direct GPT-2 model from HuggingFace

# Default values
MODEL_SIZE="small"
PROMPT="Once upon a time"
MAX_LENGTH=100
TEMPERATURE=0.7
TOP_K=50
TOP_P=0.95
REP_PENALTY=1.2
DO_SAMPLE="--do_sample"
NUM_SEQUENCES=1

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_size)
            MODEL_SIZE="$2"
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
        --rep_penalty)
            REP_PENALTY="$2"
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
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model_size <size>   GPT-2 model size to use (small, medium, large, xl) (default: small)"
            echo "  --prompt <text>       Prompt for text generation (default: 'Once upon a time')"
            echo "  --length <num>        Maximum length of generated text (default: 100)"
            echo "  --temperature <float> Temperature for sampling (default: 0.7)"
            echo "  --top_k <num>         Top-k filtering parameter (default: 50)"
            echo "  --top_p <float>       Top-p filtering parameter (default: 0.95)"
            echo "  --rep_penalty <float> Repetition penalty (default: 1.2, higher values reduce repetition)"
            echo "  --greedy              Use greedy decoding instead of sampling"
            echo "  --num_seqs <num>      Number of sequences to generate (default: 1)"
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

# Build the command
GENERATE_CMD="./direct_generation.py \
  --model_size=$MODEL_SIZE \
  --prompt=\"$PROMPT\" \
  --max_length=$MAX_LENGTH \
  --temperature=$TEMPERATURE \
  --top_k=$TOP_K \
  --top_p=$TOP_P \
  --repetition_penalty=$REP_PENALTY \
  --num_return_sequences=$NUM_SEQUENCES"

# Run the command
echo "Generating text with the following configuration:"
echo "- Model size: $MODEL_SIZE"
echo "- Prompt: \"$PROMPT\""
echo "- Max length: $MAX_LENGTH"
echo "- Temperature: $TEMPERATURE"
echo "- Top-k: $TOP_K"
echo "- Top-p: $TOP_P"
echo "- Repetition penalty: $REP_PENALTY"
echo "- Sampling: ${DO_SAMPLE:+Enabled}"
echo "- Number of sequences: $NUM_SEQUENCES"
echo ""
echo "Running command: $GENERATE_CMD"
eval $GENERATE_CMD