#!/usr/bin/env python

import torch
import os
import sys
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Optimize GPU usage and find optimal batch size")
    parser.add_argument("--max-memory-gb", type=float, default=24.0,
                       help="Maximum GPU memory in GB (default: 24.0)")
    parser.add_argument("--model-size", type=str, default="gpt2-small",
                       choices=["gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"],
                       help="Model size to optimize for")
    parser.add_argument("--start-batch-size", type=int, default=4,
                       help="Starting batch size for testing")
    parser.add_argument("--max-batch-size", type=int, default=64,
                       help="Maximum batch size to try")
    parser.add_argument("--sequence-length", type=int, default=1024,
                       help="Sequence length for testing")
    return parser.parse_args()

def optimize_gpu_settings():
    """Apply global torch optimizations"""
    if not torch.cuda.is_available():
        print("CUDA is not available. GPU optimizations won't be applied.")
        return False
    
    # Enable TensorFloat-32 (TF32) for faster matrix multiplications on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Enable cuDNN benchmark mode for optimized performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # Set environment variables for better GPU utilization
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    
    # Print GPU information
    device = torch.cuda.current_device()
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    
    return True

def find_optimal_batch_size(args):
    """Find the optimal batch size that fits in GPU memory"""
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot determine optimal batch size.")
        return None
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Get initial GPU memory usage
    initial_memory = torch.cuda.memory_allocated() / 1e9  # Convert to GB
    print(f"Initial GPU memory usage: {initial_memory:.2f} GB")
    
    # Define model dimensions based on model size
    model_dims = {
        "gpt2-small": {"d_model": 768, "num_layers": 12, "num_heads": 12},
        "gpt2-medium": {"d_model": 1024, "num_layers": 24, "num_heads": 16},
        "gpt2-large": {"d_model": 1280, "num_layers": 36, "num_heads": 20},
        "gpt2-xl": {"d_model": 1600, "num_layers": 48, "num_heads": 25}
    }
    
    dims = model_dims[args.model_size]
    print(f"Testing with model configuration: {dims}")
    
    # Memory needed per sample (empirical approximation)
    # Formula: seq_len * d_model * 4 bytes * 3 (forward, backward, optimizer) * 1.2 (overhead)
    bytes_per_token = dims["d_model"] * 4 * 3 * 1.2
    bytes_per_layer = bytes_per_token * args.sequence_length
    estimated_bytes_per_sample = bytes_per_layer * dims["num_layers"]
    
    # Convert to GB
    estimated_gb_per_sample = estimated_bytes_per_sample / 1e9
    print(f"Estimated memory per sample: {estimated_gb_per_sample:.4f} GB")
    
    # Calculate available memory
    available_gb = args.max_memory_gb - initial_memory
    print(f"Available GPU memory: {available_gb:.2f} GB")
    
    # Calculate theoretical max batch size
    theoretical_max = int(available_gb / estimated_gb_per_sample)
    theoretical_max = min(theoretical_max, args.max_batch_size)
    
    print(f"Theoretical maximum batch size: {theoretical_max}")
    
    # Test batch sizes in practice to find the largest that fits
    optimal_batch = None
    
    # Test in descending order starting from either max or theoretical_max
    for batch_size in range(theoretical_max, args.start_batch_size - 1, -1):
        try:
            print(f"Testing batch size: {batch_size}...")
            
            # Create a sample input
            input_ids = torch.randint(0, 50257, (batch_size, args.sequence_length), device="cuda")
            
            # Create a simple model matching dimensions
            model = torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(
                    d_model=dims["d_model"],
                    nhead=dims["num_heads"],
                    dim_feedforward=dims["d_model"] * 4,
                    batch_first=True
                ),
                num_layers=dims["num_layers"]
            ).cuda()
            
            # Enable mixed precision
            scaler = torch.cuda.amp.GradScaler()
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
            
            # Run a forward and backward pass with mixed precision
            with torch.cuda.amp.autocast():
                output = model(input_ids.float())
                # Fake loss
                loss = output.sum()
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # If we got here, this batch size works
            optimal_batch = batch_size
            
            # Clean up
            del model, input_ids, output
            torch.cuda.empty_cache()
            
            # Found working size, no need to test smaller ones
            break
            
        except torch.cuda.OutOfMemoryError:
            print(f"Batch size {batch_size} is too large (OOM)")
            # Clean up
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"Error testing batch size {batch_size}: {e}")
            torch.cuda.empty_cache()
            continue
    
    if optimal_batch:
        print(f"\nOptimal batch size for {args.model_size} with {args.sequence_length} sequence length: {optimal_batch}")
        print(f"Recommended gradient accumulation steps: {max(1, 32 // optimal_batch)}")
        return optimal_batch
    else:
        print("Could not determine optimal batch size. Try with a smaller starting batch size.")
        return None

def generate_training_script(batch_size, model_size, seq_length):
    """Generate an optimized training script based on findings"""
    if not batch_size:
        return False
    
    grad_accum = max(1, 32 // batch_size)
    
    script_content = f"""#!/bin/bash

# Optimized training script automatically generated
# for {model_size} with sequence length {seq_length}

# Set environment variables for better GPU performance
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=true

# Create output directory
OUTPUT_DIR="./output/optimized_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Run training with optimized parameters
python scripts/long_train.py \\
  --model_type=gpt \\
  --d_model=768 \\
  --num_heads=12 \\
  --num_layers=12 \\
  --d_ff=3072 \\
  --max_seq_length={seq_length} \\
  --dropout=0.1 \\
  --output_dir="$OUTPUT_DIR" \\
  --batch_size={batch_size} \\
  --gradient_accumulation_steps={grad_accum} \\
  --learning_rate=3e-5 \\
  --num_train_epochs=30 \\
  --warmup_ratio=0.1 \\
  --fp16 \\
  --logging_steps=50 \\
  --eval_steps=500 \\
  --save_steps=1000 \\
  --use_slimpajama \\
  --slimpajama_subset=all \\
  --metrics_file="metrics.json"

# Create a symlink to the latest training
ln -sf "$OUTPUT_DIR" ./output/latest_optimized_training

echo "Training complete. Results saved to $OUTPUT_DIR"
"""
    
    # Write the script
    with open("run_auto_optimized_training.sh", "w") as f:
        f.write(script_content)
    
    # Make executable
    os.chmod("run_auto_optimized_training.sh", 0o755)
    
    print("\nCreated optimized training script: run_auto_optimized_training.sh")
    print(f"The script uses batch_size={batch_size} and gradient_accumulation_steps={grad_accum}")
    return True

def main():
    args = parse_args()
    
    print("=== GPU Optimization Tool ===")
    
    if not optimize_gpu_settings():
        print("Failed to optimize GPU settings. Exiting.")
        return
    
    print("\n=== Finding Optimal Batch Size ===")
    optimal_batch = find_optimal_batch_size(args)
    
    if optimal_batch:
        print("\n=== Generating Optimized Training Script ===")
        if generate_training_script(optimal_batch, args.model_size, args.sequence_length):
            print("\nDone! Run ./run_auto_optimized_training.sh to start training with optimal parameters.")
    
if __name__ == "__main__":
    main()