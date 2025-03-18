# Simple GPT

A Python implementation of a transformer-based language model inspired by GPT (Generative Pre-trained Transformer) architecture. This project aims to provide a clear and educational implementation of modern transformer models for text generation.

## Features

- Implementation of transformer models with multi-head attention
- Support for both encoder-decoder transformer and decoder-only GPT models
- Training with mixed precision and gradient accumulation
- Integration with the Hugging Face ecosystem (tokenizers, datasets)
- Text generation with various sampling strategies
- Optional Weights & Biases integration for experiment tracking

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/simple_gpt.git
cd simple_gpt

# Install the package
pip install -e .
```

## Quick Start

### Training a model

```bash
python -m simple_gpt.scripts.train \
    --model_type=gpt \
    --dataset_name=wikitext \
    --dataset_config_name=wikitext-103-raw-v1 \
    --num_train_epochs=3 \
    --output_dir=./output/my_gpt_model
```

### Generating text

```bash
python -m simple_gpt.scripts.generate \
    --model_path=./output/my_gpt_model \
    --prompt="In a world where AI had become sentient, " \
    --max_length=200 \
    --do_sample \
    --temperature=0.8
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

## Extended Training

For longer training runs with more data, you can use the extended training script:

```bash
python scripts/train_extended.py \
    --model_type=gpt \
    --d_model=512 \
    --num_heads=8 \
    --num_layers=8 \
    --dataset_name=wikitext \
    --dataset_config_name=wikitext-103-raw-v1 \
    --max_train_samples=10000 \
    --num_train_epochs=10 \
    --output_dir=./output/long_training_model
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
