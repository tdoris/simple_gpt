#!/usr/bin/env python

import os
import json
import torch
import logging
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from simple_gpt.configs import ModelConfig
from simple_gpt.models import GPTModel

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fetch GPT-2 weights from HuggingFace")
    
    parser.add_argument("--model_size", type=str, default="small", 
                        choices=["small", "medium", "large", "xl"],
                        help="GPT-2 model size to download")
    parser.add_argument("--output_dir", type=str, default="./output/gpt2_weights",
                        help="Directory to save the model")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Map model size to HuggingFace model name and configuration
    model_map = {
        "small": {
            "name": "gpt2",
            "config": {
                "d_model": 768,
                "num_heads": 12,
                "num_layers": 12,
                "d_ff": 3072
            }
        },
        "medium": {
            "name": "gpt2-medium",
            "config": {
                "d_model": 1024,
                "num_heads": 16,
                "num_layers": 24,
                "d_ff": 4096
            }
        },
        "large": {
            "name": "gpt2-large",
            "config": {
                "d_model": 1280,
                "num_heads": 20,
                "num_layers": 36,
                "d_ff": 5120
            }
        },
        "xl": {
            "name": "gpt2-xl",
            "config": {
                "d_model": 1600,
                "num_heads": 25,
                "num_layers": 48,
                "d_ff": 6400
            }
        }
    }
    
    model_info = model_map[args.model_size]
    hf_model_name = model_info["name"]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Fetching {hf_model_name} weights from HuggingFace...")
    
    # Set cache directories
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(args.output_dir)), "cache")
    hf_cache_dir = os.path.join(cache_dir, "huggingface")
    
    # Ensure cache directories exist
    os.makedirs(hf_cache_dir, exist_ok=True)
    
    # Set environment variables for cache
    os.environ['HF_HOME'] = hf_cache_dir
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(hf_cache_dir, "transformers")
    
    logger.info(f"Using cache directory: {hf_cache_dir}")
    
    # Download tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(hf_model_name, cache_dir=hf_cache_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Download the model
    hf_model = GPT2LMHeadModel.from_pretrained(hf_model_name, cache_dir=hf_cache_dir)
    
    # Initialize our GPTModel with the right configuration
    config = model_info["config"]
    max_seq_length = 1024  # Default GPT-2 sequence length
    
    model_config = ModelConfig(
        model_type="gpt",
        vocab_size=len(tokenizer),
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        max_seq_length=max_seq_length,
        dropout=0.1  # Default dropout
    )
    
    # Save model configuration
    model_config_dict = {
        "model_type": "gpt",
        "vocab_size": len(tokenizer),
        "d_model": config["d_model"],
        "num_heads": config["num_heads"],
        "num_layers": config["num_layers"],
        "d_ff": config["d_ff"],
        "max_seq_length": max_seq_length,
        "dropout": 0.1
    }
    
    with open(os.path.join(args.output_dir, "model_config.json"), "w") as f:
        json.dump(model_config_dict, f, indent=2)
    
    # Extract and convert weights
    # We need to map HuggingFace GPT-2 weights to our model's format
    # First, let's see what we have in the HuggingFace model
    hf_state_dict = hf_model.state_dict()
    
    # Print some diagnostic information
    logger.info("Analyzing GPT-2 weights for conversion")
    key_sample = "transformer.h.0.attn.c_attn.weight"
    logger.info(f"Shape of {key_sample}: {hf_state_dict[key_sample].shape}")
    logger.info(f"Type of {key_sample}: {hf_state_dict[key_sample].dtype}")
    
    # Create a new state dict for our model
    state_dict = {}
    
    # Map the word embeddings
    state_dict["token_embedding.weight"] = hf_state_dict["transformer.wte.weight"]
    state_dict["position_embedding.weight"] = hf_state_dict["transformer.wpe.weight"]
    state_dict["lm_head.weight"] = hf_state_dict["transformer.wte.weight"]  # Weight tying
    state_dict["norm.weight"] = hf_state_dict["transformer.ln_f.weight"]
    state_dict["norm.bias"] = hf_state_dict["transformer.ln_f.bias"]
    
    # Map each layer
    for i in range(config["num_layers"]):
        hf_prefix = f"transformer.h.{i}."
        our_prefix = f"layers.{i}."
        
        # Layer normalization 1
        state_dict[our_prefix + "norm1.weight"] = hf_state_dict[hf_prefix + "ln_1.weight"]
        state_dict[our_prefix + "norm1.bias"] = hf_state_dict[hf_prefix + "ln_1.bias"]
        
        # Layer normalization 2
        state_dict[our_prefix + "norm2.weight"] = hf_state_dict[hf_prefix + "ln_2.weight"]
        state_dict[our_prefix + "norm2.bias"] = hf_state_dict[hf_prefix + "ln_2.bias"]
        
        # Self-attention weights - GPT-2 uses a combined QKV projection
        # We'll need to split it for our model
        c_attn_weight = hf_state_dict[hf_prefix + "attn.c_attn.weight"]  # [d_model, 3*d_model]
        c_attn_bias = hf_state_dict[hf_prefix + "attn.c_attn.bias"]  # [3*d_model]
        
        # For GPT-2, the attention weights shape is (d_model, 3*d_model)
        # The bias shape is (3*d_model)
        # We need to split into query, key, value components
        d_model = config["d_model"]
        
        # For GPT-2, attention weights are [d_model, 3*d_model]
        # We need to transpose and split these properly
        
        # Diagnostics for each layer to understand the shapes
        if i == 0:
            logger.info(f"Layer 0 attention shapes before split:")
            logger.info(f"  c_attn_weight: {c_attn_weight.shape}")
            logger.info(f"  c_attn_bias: {c_attn_bias.shape}")
        
        # Split into query, key, value sections
        split_size = d_model
        w_splits = torch.split(c_attn_weight, split_size, dim=1)
        b_splits = torch.split(c_attn_bias, split_size)
        
        # Get the individual parts
        qw, kw, vw = w_splits
        qb, kb, vb = b_splits
        
        # Our model expects these in transposed form
        qw = qw.transpose(0, 1)
        kw = kw.transpose(0, 1)  
        vw = vw.transpose(0, 1)
        
        # Store query, key, value weights and biases
        state_dict[our_prefix + "self_attention.query.weight"] = qw
        state_dict[our_prefix + "self_attention.query.bias"] = qb
        state_dict[our_prefix + "self_attention.key.weight"] = kw
        state_dict[our_prefix + "self_attention.key.bias"] = kb
        state_dict[our_prefix + "self_attention.value.weight"] = vw
        state_dict[our_prefix + "self_attention.value.bias"] = vb
        
        # Output projection - also needs transposition
        # GPT-2 shape is typically [d_model, d_model]
        c_proj_weight = hf_state_dict[hf_prefix + "attn.c_proj.weight"]
        c_proj_bias = hf_state_dict[hf_prefix + "attn.c_proj.bias"]
        
        if i == 0:
            logger.info(f"  c_proj_weight: {c_proj_weight.shape}")
            logger.info(f"  c_proj_bias: {c_proj_bias.shape}")
        
        # Transpose the projection weight to match our model's expected shape
        state_dict[our_prefix + "self_attention.output_layer.weight"] = c_proj_weight.transpose(0, 1)
        state_dict[our_prefix + "self_attention.output_layer.bias"] = c_proj_bias
        
        # Feed forward - transpose the weights to match our model's expectations
        # In our model: fc1.weight: [d_ff, d_model], fc2.weight: [d_model, d_ff]
        # In GPT-2: mlp.c_fc.weight: [d_model, d_ff], mlp.c_proj.weight: [d_ff, d_model]
        
        # Transpose feed forward layers
        state_dict[our_prefix + "feed_forward.fc1.weight"] = hf_state_dict[hf_prefix + "mlp.c_fc.weight"].transpose(0, 1)
        state_dict[our_prefix + "feed_forward.fc1.bias"] = hf_state_dict[hf_prefix + "mlp.c_fc.bias"]
        state_dict[our_prefix + "feed_forward.fc2.weight"] = hf_state_dict[hf_prefix + "mlp.c_proj.weight"].transpose(0, 1)
        state_dict[our_prefix + "feed_forward.fc2.bias"] = hf_state_dict[hf_prefix + "mlp.c_proj.bias"]
    
    # Save the converted state dict
    torch.save(state_dict, os.path.join(args.output_dir, "pytorch_model.bin"))
    
    logger.info(f"Successfully saved {hf_model_name} model to {args.output_dir}")
    logger.info(f"Model architecture: {config['num_layers']} layers, {config['d_model']} dimensions, {config['num_heads']} attention heads")
    logger.info(f"Vocabulary size: {len(tokenizer)}")
    
    logger.info("\nTo generate text with this model, use:")
    logger.info(f"python -m simple_gpt.scripts.generate --model_path={args.output_dir} --prompt=\"Your prompt here\" --max_length=100 --temperature=0.7 --do_sample")

if __name__ == "__main__":
    main()