#!/bin/bash

# Model Comparison Script
# This script runs a comparison between two language models using Claude as the evaluator

# Default values
MODEL_A="Default-Model-A"
MODEL_B="Default-Model-B"
PROMPTS_PER_CATEGORY=10
CATEGORIES=5
OUTPUT_FILE=""

# Function to show usage
show_usage() {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  --model-a <name>          Name of the first model (required)"
  echo "  --model-b <name>          Name of the second model (required)"
  echo "  --prompts <number>        Number of prompts per category (default: 10, max: 10)"
  echo "  --categories <number>     Number of categories to test (default: 5, max: 5)"
  echo "  --output <file>           Path to save the JSON results (default: auto-generated)"
  echo "  --help                    Show this help message"
  echo ""
  echo "Environment variables:"
  echo "  ANTHROPIC_API_KEY         Required for Claude evaluation"
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model-a)
      MODEL_A="$2"
      shift 2
      ;;
    --model-b)
      MODEL_B="$2"
      shift 2
      ;;
    --prompts)
      PROMPTS_PER_CATEGORY="$2"
      shift 2
      ;;
    --categories)
      CATEGORIES="$2"
      shift 2
      ;;
    --output)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    --help)
      show_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      show_usage
      exit 1
      ;;
  esac
done

# Check for required arguments
if [ "$MODEL_A" == "Default-Model-A" ] || [ "$MODEL_B" == "Default-Model-B" ]; then
  echo "Error: Both --model-a and --model-b are required."
  show_usage
  exit 1
fi

# Check for Anthropic API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "Error: ANTHROPIC_API_KEY environment variable is not set."
  echo "Please set it with: export ANTHROPIC_API_KEY=your_key_here"
  exit 1
fi

# Build the command
if [ -n "$OUTPUT_FILE" ]; then
  OUTPUT_ARG="--output $OUTPUT_FILE"
else
  OUTPUT_ARG=""
fi

COMMAND="python model_comparison.py --model-a \"$MODEL_A\" --model-b \"$MODEL_B\" --prompts-per-category $PROMPTS_PER_CATEGORY --categories $CATEGORIES $OUTPUT_ARG"

# Add a note about models being tested
echo "Note: This script has been modified to work with local SimpleGPT models."
echo "It calls the generate.py script to generate responses from the models."

# Display information
echo "Running model comparison with the following configuration:"
echo "- Model A: $MODEL_A"
echo "- Model B: $MODEL_B"
echo "- Prompts per category: $PROMPTS_PER_CATEGORY"
echo "- Number of categories: $CATEGORIES"
if [ -n "$OUTPUT_FILE" ]; then
  echo "- Output file: $OUTPUT_FILE"
else
  echo "- Output file: auto-generated"
fi
echo ""

# Execute the comparison
echo "Starting comparison..."
eval $COMMAND