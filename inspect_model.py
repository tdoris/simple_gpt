#!/usr/bin/env python

import os
import argparse
import json
import torch
from transformers import GPT2Tokenizer
from simple_gpt.configs import ModelConfig
from simple_gpt.trainers import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect a trained model and its configuration")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model directory")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed information about the model")
    return parser.parse_args()


def get_model_info(model_path, verbose=False):
    """Get information about a trained model."""
    info = {"model_path": model_path}
    
    # Check model config
    model_config_path = os.path.join(model_path, "model_config.json")
    if os.path.exists(model_config_path):
        with open(model_config_path, "r") as f:
            model_config = json.load(f)
        info["model_config"] = model_config
    
    # Check training config
    training_config_path = os.path.join(model_path, "training_config.json")
    if os.path.exists(training_config_path):
        with open(training_config_path, "r") as f:
            training_config = json.load(f)
        info["training_config"] = training_config
    
    # Check if there's a metrics file
    metrics_path = os.path.join(model_path, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        if metrics and len(metrics) > 0:
            last_metric = metrics[-1]
            info["last_metrics"] = {
                "step": last_metric.get("step"),
                "loss": last_metric.get("loss"),
                "learning_rate": last_metric.get("learning_rate"),
                "epoch": last_metric.get("epoch"),
            }
            if "perplexity" in last_metric:
                info["last_metrics"]["perplexity"] = last_metric["perplexity"]
    
    # Check args.json if it exists
    args_path = os.path.join(model_path, "args.json")
    if os.path.exists(args_path):
        with open(args_path, "r") as f:
            args = json.load(f)
        info["args"] = args
    
    # Check if model checkpoint exists
    model_bin_path = os.path.join(model_path, "pytorch_model.bin")
    info["model_file_exists"] = os.path.exists(model_bin_path)
    
    # Check tokenizer files
    tokenizer_files = ["vocab.json", "merges.txt", "tokenizer_config.json"]
    info["tokenizer_files_exist"] = all(os.path.exists(os.path.join(model_path, f)) for f in tokenizer_files)
    
    # Check for checkpoint directories
    checkpoints = []
    for item in os.listdir(model_path):
        if item.startswith("checkpoint-") and os.path.isdir(os.path.join(model_path, item)):
            checkpoints.append(item)
    info["checkpoints"] = sorted(checkpoints)
    
    return info


def print_model_info(info, verbose=False):
    """Print information about a trained model in a formatted way."""
    print("\n" + "="*50)
    print(f"Model path: {info['model_path']}")
    print("="*50)
    
    if "model_config" in info:
        model_config = info["model_config"]
        print("\nModel Configuration:")
        print(f"  Model type: {model_config.get('model_type', 'Unknown')}")
        print(f"  Dimensions: d_model={model_config.get('d_model', 'Unknown')}, d_ff={model_config.get('d_ff', 'Unknown')}")
        print(f"  Attention heads: {model_config.get('num_heads', 'Unknown')}")
        print(f"  Layers: {model_config.get('num_layers', 'Unknown')}")
        print(f"  Vocab size: {model_config.get('vocab_size', 'Unknown')}")
        print(f"  Max sequence length: {model_config.get('max_seq_length', 'Unknown')}")
    
    if "last_metrics" in info:
        metrics = info["last_metrics"]
        print("\nTraining Metrics:")
        print(f"  Last step: {metrics.get('step', 'Unknown')}")
        print(f"  Last loss: {metrics.get('loss', 'Unknown'):.4f}")
        if "perplexity" in metrics:
            print(f"  Perplexity: {metrics.get('perplexity', 'Unknown'):.2f}")
        print(f"  Epoch: {metrics.get('epoch', 'Unknown'):.2f}")
    
    print("\nModel Files:")
    print(f"  Model checkpoint: {'✓' if info.get('model_file_exists', False) else '✗'}")
    print(f"  Tokenizer files: {'✓' if info.get('tokenizer_files_exist', False) else '✗'}")
    
    if "checkpoints" in info and info["checkpoints"]:
        print(f"\nCheckpoints available: {len(info['checkpoints'])}")
        if verbose:
            for checkpoint in info["checkpoints"]:
                print(f"  - {checkpoint}")
    
    if verbose and "args" in info:
        print("\nTraining Arguments:")
        args = info["args"]
        for key, value in args.items():
            print(f"  {key}: {value}")
    
    print("\nTo use this model for text generation, run:")
    print(f"  ./run_text_generation.sh \"{info['model_path']}\" \"Your prompt here\"")


def load_and_verify_model(model_path, verbose=False):
    """Load the model to verify it works properly."""
    info = get_model_info(model_path, verbose)
    
    # Create model config based on model_config.json if available
    model_config_dict = info.get("model_config", {})
    model_config = ModelConfig(
        model_type=model_config_dict.get("model_type", "gpt"),
        vocab_size=model_config_dict.get("vocab_size", 50257),
        d_model=model_config_dict.get("d_model", 768),
        num_heads=model_config_dict.get("num_heads", 12),
        num_layers=model_config_dict.get("num_layers", 12),
        d_ff=model_config_dict.get("d_ff", 3072),
        max_seq_length=model_config_dict.get("max_seq_length", 1024),
    )
    
    # Try to load the model
    try:
        model = Trainer.load_model(model_path, model_config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print("\nModel loaded successfully! ✓")
        
        # Get model size
        model_size = sum(p.numel() for p in model.parameters()) / 1_000_000
        print(f"Model parameters: {model_size:.2f}M")
        
        # Try to load the tokenizer
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(
                model_path if os.path.exists(os.path.join(model_path, "vocab.json")) else "gpt2"
            )
            print("Tokenizer loaded successfully! ✓")
            
            # Verify model works with a simple forward pass
            dummy_input = torch.tensor([[1, 2, 3, 4, 5]]).to(device)
            with torch.no_grad():
                outputs = model(dummy_input)
            print("Model forward pass successful! ✓")
        except Exception as e:
            print(f"Failed to load tokenizer: {e}")
    except Exception as e:
        print(f"\nFailed to load model: {e}")
    
    return info


def main():
    args = parse_args()
    
    # Load and verify the model
    info = load_and_verify_model(args.model_path, args.verbose)
    
    # Print information about the model
    print_model_info(info, args.verbose)


if __name__ == "__main__":
    main()