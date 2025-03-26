# Model Comparison Tool

This tool allows you to compare the performance of two language models using Claude 3.5 Sonnet as an evaluator. The comparison is based on a diverse set of prompts across five different categories, with Claude ranking each pair of responses.

## Features

- Compares two models across 5 different prompt categories:
  - Factual Knowledge
  - Reasoning & Problem Solving
  - Creative Writing
  - Summarization & Analysis
  - Ethical Reasoning
  
- Uses Claude 3.5 Sonnet as an unbiased evaluator
- Generates a comprehensive scorecard with overall and per-category results
- Saves detailed evaluation results in JSON format
- Includes sample evaluations for each category

## Requirements

- Python 3.7+
- Anthropic API key
- `anthropic` Python package

Install the required package:

```
pip install anthropic
```

## Usage

### Basic Usage

1. Set your Anthropic API key as an environment variable:

```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

2. Run the comparison using the shell script:

```bash
./run_model_comparison.sh --model-a "Model-A-Name" --model-b "Model-B-Name"
```

### Command Line Options

```
Usage: ./run_model_comparison.sh [options]

Options:
  --model-a <name>          Name of the first model (required)
  --model-b <name>          Name of the second model (required)
  --prompts <number>        Number of prompts per category (default: 10, max: 10)
  --output <file>           Path to save the JSON results (default: auto-generated)
  --help                    Show this help message
```

## Example Output

The tool generates a comprehensive report like this:

```
================================================================================
MODEL COMPARISON REPORT: GPT-4o vs Claude-3-Opus
Date: 2024-07-15 14:30:25
================================================================================

OVERALL RESULTS:
--------------------------------------------------------------------------------
GPT-4o: 27.5/50 (55.0%)
Claude-3-Opus: 22.5/50 (45.0%)

🏆 WINNER: GPT-4o (by 5.0 points)

RESULTS BY CATEGORY:
--------------------------------------------------------------------------------

Creative Writing (based on 10 prompts):
  GPT-4o: 7.0/10 (70.0%)
  Claude-3-Opus: 3.0/10 (30.0%)
  Category Winner: GPT-4o

Ethical Reasoning (based on 10 prompts):
  GPT-4o: 4.0/10 (40.0%)
  Claude-3-Opus: 6.0/10 (60.0%)
  Category Winner: Claude-3-Opus

...
```

## Customization

### Adding New Prompts

You can modify the `PROMPT_CATEGORIES` dictionary in `model_comparison.py` to add or change prompts.

### Implementing Model API Calls

Currently, the script contains placeholder functions for getting model responses. To use with actual models:

1. Modify the `get_model_response()` method in the `ModelComparer` class
2. Implement the actual API calls to your models

## Notes

- The tool uses a random ordering of responses to avoid positional bias
- You can run with fewer prompts to get faster results
- The JSON output contains all detailed evaluations for further analysis
- Claude evaluates based on accuracy, clarity, helpfulness, depth, and ethical considerations