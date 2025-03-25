# SimpleGPT - Generic Language Model Training

A Python implementation of a transformer-based language model inspired by GPT (Generative Pre-trained Transformer) architecture. This project provides a clear and educational implementation of modern transformer models for text generation with a generic, dataset-agnostic training system.

## Features

- Implementation of transformer models with multi-head attention
- Support for both encoder-decoder transformer and decoder-only GPT models
- Generic dataset-agnostic training system supporting any HuggingFace dataset
- Training with mixed precision and gradient accumulation
- Integration with the Hugging Face ecosystem (tokenizers, datasets)
- Enhanced caching mechanisms for offline and local file usage
- Text generation with various sampling strategies
- Optional Weights & Biases integration for experiment tracking
- Robust training with automatic hang detection and recovery
- Comprehensive testing framework for validating configurations
- Visualization tools for analyzing training results

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/simple_gpt.git
cd simple_gpt

# Install the package
pip install -e .
```

## Quick Start

### Testing the Generic Training System

To verify dataset loading and training with the generic system:

```bash
# Test dataset loading for wikitext
./test_dataset.sh

# Test dataset loading for Simple Wikipedia
./test_dataset.sh --use-simple-wikipedia

# Run a quick test of multiple configurations
./test_all_configs.sh --quick
```

For full documentation on the testing framework:
```bash
# Display detailed documentation
cat README_TESTING.md
```

### Starting Generic Training

```bash
# Train on wikitext dataset
./run_long_training.sh

# Train on Simple Wikipedia dataset
./run_long_training.sh --use-simple-wikipedia

# Train with limited samples (for testing)
./run_long_training.sh --train-samples 1000 --val-samples 100 --epochs 1
```

### Testing Training Pipeline

To quickly verify the basic training pipeline works correctly:

```bash
./test_long_training.sh
```

This will train a small GPT model on a subset of WikiText and generate sample text.

### Extended Training with Monitoring

For robust training with automatic hang detection:

```bash
./start_and_monitor.sh
```

By default, this uses the entire WikiText dataset. To limit training to a subset:

```bash
MAX_TRAIN_SAMPLES=50000 ./start_and_monitor.sh
```

This script will:
1. Start the long training process
2. Monitor for any training hangs
3. Automatically restart training if it detects issues
4. Create checkpoints during training

### Generating Text from a Trained Model

```bash
./run_text_generation.sh [options]
```

Options:
- `--model <path>`: Path to the model directory (default: ./output/slimpajama_arxiv_20250324_084707)
- `--prompt <text>`: Prompt for text generation (default: "Once upon a time")
- `--length <n>`: Maximum length of generated text (default: 100)
- `--temperature <t>`: Temperature for sampling (default: 0.6)
- `--top_k <n>`: Top-k filtering parameter (default: 40)
- `--top_p <p>`: Top-p nucleus sampling parameter (default: 0.92)
- `--greedy`: Use greedy decoding instead of sampling
- `--num_seqs <n>`: Number of sequences to generate (default: 1)
- `--rep_penalty <float>`: Repetition penalty (default: 1.5)
- `--min_length <n>`: Minimum length of generated text (default: 0)

Notes:
- For best results, use the SlimPajama ArXiv model (default)
- Lower temperature (0.4-0.6) produces more coherent text
- Higher repetition penalty (1.5-1.8) helps avoid repetitive tokens
- Consider using greedy decoding for short factual completions

Example:
```bash
./run_text_generation.sh --model ./output/latest_extended_training --prompt "Once upon a time" --length 200 --temperature 0.7
```

### Using Pre-trained GPT-2 Models

You can download the pre-trained GPT-2 models from HuggingFace and use them directly:

```bash
# Download pre-trained GPT-2 weights
./fetch_gpt2_weights.py --model_size small --output_dir ./output/gpt2_weights

# Generate text using the pre-trained model
./run_text_generation.sh --model ./output/gpt2_weights --prompt "Once upon a time"
```

Available model sizes:
- `small`: 124M parameters (default)
- `medium`: 355M parameters
- `large`: 774M parameters
- `xl`: 1.5B parameters

### Inspecting Models

To inspect a trained model:

```bash
python inspect_model.py --model_path <model_path> [--verbose]
```

## Model Architecture

The project implements two main transformer architectures:

1. **TransformerModel**: A full encoder-decoder transformer similar to the original "Attention is All You Need" paper.
2. **GPTModel**: A decoder-only transformer similar to the GPT architecture, suitable for autoregressive text generation.

Both models use the following key components:
- Multi-head self-attention mechanism
- Position-wise feed-forward networks
- Layer normalization
- Positional encodings

## Configuration

The behavior of the models and training process can be customized through configuration classes:

- `ModelConfig`: Configure model architecture (layers, dimensions, etc.)
- `TrainingConfig`: Configure training hyperparameters (learning rate, batch size, etc.)
- `DataConfig`: Configure dataset loading and preprocessing
- `GenerationConfig`: Configure text generation parameters (temperature, sampling, etc.)

## Training Scripts

### Basic Training

```bash
python -m simple_gpt.scripts.train \
    --model_type=gpt \
    --dataset_name=wikitext \
    --dataset_config_name=wikitext-103-raw-v1 \
    --num_train_epochs=3 \
    --output_dir=./output/my_gpt_model
```

### Extended Training

For longer training runs, use the long_train.py script which now defaults to using the entire dataset:

```bash
python scripts/long_train.py \
    --model_type=gpt \
    --d_model=768 \
    --num_heads=12 \
    --num_layers=12 \
    --d_ff=3072 \
    --max_seq_length=512 \
    --dropout=0.1 \
    --output_dir=./output/extended_training \
    --batch_size=4 \
    --gradient_accumulation_steps=8 \
    --learning_rate=3e-5 \
    --num_train_epochs=10 \
    --max_train_time=12 \
    --warmup_ratio=0.1 \
    --fp16 \
    --dataset_name=wikitext \
    --dataset_config_name=wikitext-103-raw-v1
```

If you want to limit training to a subset of the data, use the `--max_train_samples` option:

```bash
python scripts/long_train.py --max_train_samples=50000
```

### Training with SlimPajama-30B Dataset

You can train on the SlimPajama-30B dataset, which is a filtered, deduplicated subset of the Pile:

```bash
./run_slimpajama_training.sh --subset all --epochs 30
```

The script defaults to using the entire dataset. If you want to limit training to a subset:

```bash
./run_slimpajama_training.sh --subset all --train-samples 100000 --epochs 30
```

**Note:** Access to the SlimPajama dataset requires:
1. Hugging Face authentication (run `huggingface-cli login` first)
2. Acceptance of the dataset terms of use on the Hugging Face Hub

If these requirements are not met, the script will automatically fall back to using the WikiText dataset.

Available options for the SlimPajama subsets:
- `all`: Use a balanced mix across all domains
- `arxiv`: Research papers from arXiv
- `github`: Code from GitHub repositories
- `stackexchange`: Q&A from Stack Exchange
- `wiki`: Wikipedia articles
- `books`: Books from various sources
- `c4`: Subset of the Common Crawl corpus
- `common_crawl`: Web pages from Common Crawl

### Continuing Training from a Checkpoint

To continue training from a saved checkpoint with the generic system:

```bash
# Continue training with default dataset (wikitext)
./continue_training_generic.sh --model-path ./output/my_model --epochs 3

# Continue training with Simple Wikipedia dataset
./continue_training_generic.sh --model-path ./output/my_model --use-simple-wikipedia

# Continue with optimized GPU parameters
./continue_training_generic.sh --model-path ./output/my_model --batch-size 16 --grad-accum 4 --seq-length 1024
```

This allows you to:
- Resume training that was interrupted
- Train a pre-trained model on new data
- Fine-tune an existing model with different parameters
- Switch datasets during training

By default, the script will use the entire dataset. If you want to limit training to a subset, use the `--train-samples` option.

Available options:
```
--model-path <path>         Path to the model checkpoint (required)
--output-dir <path>         Directory to save the continued model
--epochs <num>              Number of training epochs (default: 1000)
--max-time <hours>          Maximum training time in hours (default: 0 = no limit)
--train-samples <num>       Number of training samples (default: all samples)
--val-samples <num>         Number of validation samples (default: all samples)
--batch-size <num>          Batch size (default: 16)
--grad-accum <num>          Gradient accumulation steps (default: 4)
--seq-length <num>          Maximum sequence length (default: 1024)
--learning-rate <num>       Learning rate (default: 3e-5)
--dataset <name>            Dataset name (default: wikitext)
--dataset-config <name>     Dataset configuration (default: wikitext-103-raw-v1)
--use-simple-wikipedia      Use the Simple Wikipedia dataset
--offline-mode              Use offline mode (only use local files)
```

To continue with the original (legacy) training script:

```bash
./continue_training.sh --model-path ./output/my_model --epochs 3
```

To use SlimPajama with the main training script:

```bash
python scripts/long_train.py \
    --model_type=gpt \
    --d_model=768 \
    --num_heads=12 \
    --num_layers=12 \
    --d_ff=3072 \
    --max_seq_length=512 \
    --dropout=0.1 \
    --output_dir=./output/slimpajama_training \
    --batch_size=4 \
    --gradient_accumulation_steps=8 \
    --learning_rate=3e-5 \
    --num_train_epochs=30 \
    --max_train_time=0 \
    --warmup_ratio=0.1 \
    --fp16 \
    --use_slimpajama \
    --slimpajama_subset=arxiv \
    --max_train_samples=100000
```

For debugging issues with training, add the `--debug_mode` flag.

## Advanced Usage

For advanced usage, you can import the components directly in your Python code:

```python
from simple_gpt.models import GPTModel
from simple_gpt.utils import load_tokenizer
from simple_gpt.trainers import Trainer
from simple_gpt.configs import ModelConfig

# Create a custom model
model = GPTModel(
    vocab_size=50257,  # GPT-2 vocabulary size
    d_model=768,
    num_heads=12,
    num_layers=12
)

# Load a tokenizer
tokenizer = load_tokenizer()

# Use the trainer for custom training loops
trainer = Trainer(model, train_dataloader, eval_dataloader)
trainer.train()

# Load a trained model
model_config = ModelConfig(model_type="gpt")
loaded_model = Trainer.load_model("./output/my_model", model_config)
```

## Model Saving and Loading

The trainer automatically saves the model architecture configuration alongside the weights, making it easier to load models correctly. When saving a model:

```python
trainer.save_model(output_dir)
```

This will create:
- `pytorch_model.bin`: The model weights
- `training_config.json`: Configuration used for training
- `model_config.json`: Architecture parameters (dimensions, layers, etc.)

When loading a model, you can provide default parameters that will be overridden by the saved configuration if available:

```python
from simple_gpt.configs import ModelConfig
from simple_gpt.trainers import Trainer

# Create a default model config
model_config = ModelConfig()

# Load the model - will automatically use parameters from model_config.json if it exists
model = Trainer.load_model("./output/my_model", model_config)
```

## Training Monitoring

### Real-time Monitoring

During training, you can monitor the progress using:

```bash
python scripts/monitor_training.py --metrics_file <path_to_metrics_file>
```

This will show a real-time display of:
- Training loss
- Evaluation metrics
- GPU usage
- Estimated time remaining

### Configuration Testing and Visualization

The generic training system includes comprehensive testing and visualization tools:

```bash
# Run tests for all configurations
./test_all_configs.sh

# Visualize the results
./visualize_test_results.py --test-dir ./output/config_tests_*

# Compare different dimensions
./visualize_test_results.py --test-dir ./output/config_tests_* --compare dataset
./visualize_test_results.py --test-dir ./output/config_tests_* --compare batch
./visualize_test_results.py --test-dir ./output/config_tests_* --compare grad_accum
```

The visualization tools generate:
1. **Loss comparison plots** showing training loss trends across different configurations
2. **Configuration heatmaps** visualizing which parameter combinations work best
3. **Perplexity plots** comparing evaluation metrics

For full documentation on the testing and visualization framework:
```bash
cat README_TESTING.md
```

## Troubleshooting

### Training Hangs

If training seems to hang, you can try:

1. Use the `start_and_monitor.sh` script which automatically detects and restarts hanged training.
2. Reduce batch size or model dimensions if you're running out of memory.
3. Enable `--debug_mode` to get more verbose logging about what's happening.

### GPU Memory Issues

If you encounter GPU memory issues:

1. Reduce `--batch_size` and increase `--gradient_accumulation_steps` to maintain effective batch size.
2. Reduce model size (d_model, num_layers, num_heads).
3. Reduce sequence length with `--max_seq_length`.
4. Enable mixed precision training with `--fp16`.

## Documentation

The project includes comprehensive documentation:

- [README_TESTING.md](README_TESTING.md): Detailed guide for testing the generic training system
- [README_GENERIC.md](README_GENERIC.md): Documentation for the generic dataset-agnostic training system
- [OPTIMIZED_TRAINING_GENERIC.md](OPTIMIZED_TRAINING_GENERIC.md): Guide for optimized GPU training parameters

## License

This project is licensed under the MIT License - see the LICENSE file for details.