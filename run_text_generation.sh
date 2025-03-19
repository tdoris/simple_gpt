#!/bin/bash

# Script to generate text using a trained GPT model

# Default values
MODEL_PATH=""
PROMPT=""
MAX_LENGTH=100
TEMPERATURE=0.8
TOP_P=0.9
NUM_SAMPLES=1

# Parse command line arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <model_path> <prompt> [--max_length MAX_LENGTH] [--temperature TEMP] [--top_p TOP_P] [--num_samples NUM]"
    echo "Example: $0 ./output/latest_extended_training \"In the beginning\" --max_length 200"
    exit 1
fi

MODEL_PATH="$1"
PROMPT="$2"
shift 2

# Parse additional options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max_length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top_p)
            TOP_P="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if the model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path '$MODEL_PATH' does not exist."
    exit 1
fi

# Set environment variables for cache directories
export HF_HOME="./cache/huggingface"
export HF_DATASETS_CACHE="./cache/datasets"

# Run the generation script
python -c "
import os
import sys
import json
import torch
from transformers import GPT2Tokenizer
from simple_gpt.configs import ModelConfig
from simple_gpt.trainers import Trainer

def generate_text(model_path, prompt, max_length=100, temperature=0.8, top_p=0.9, num_samples=1):
    # Load model configuration
    model_config_path = os.path.join(model_path, 'model_config.json')
    if not os.path.exists(model_config_path):
        print(f'Error: Model config file not found at {model_config_path}')
        return

    # Create a directory for generation output
    output_dir = os.path.join(model_path, 'generation')
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load model
        print('Loading model...')
        
        # Load model config from JSON
        with open(model_config_path, 'r') as f:
            model_config_dict = json.load(f)
            
        # Create model config
        model_config = ModelConfig(
            model_type=model_config_dict.get('model_type', 'gpt'),
            vocab_size=model_config_dict.get('vocab_size', 50257),
            d_model=model_config_dict.get('d_model', 768),
            num_heads=model_config_dict.get('num_heads', 12),
            num_layers=model_config_dict.get('num_layers', 12),
            d_ff=model_config_dict.get('d_ff', 3072),
            max_seq_length=model_config_dict.get('max_seq_length', 1024),
            dropout=model_config_dict.get('dropout', 0.1),
        )
        
        model = Trainer.load_model(model_path, model_config)
        
        # Move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        print(f'Model loaded successfully! Using device: {device}')
        
        # Load tokenizer
        print('Loading tokenizer...')
        tokenizer_path = model_path if os.path.exists(os.path.join(model_path, 'vocab.json')) else 'gpt2'
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        print('Tokenizer loaded successfully!')
        
        # Generate text
        print(f'Generating {num_samples} text samples with parameters:')
        print(f'  Max length: {max_length}')
        print(f'  Temperature: {temperature}')
        print(f'  Top p: {top_p}')
        print(f'Input prompt: {prompt}')
        print('\\nGenerating...')
        
        # Encode prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        for i in range(num_samples):
            # Generate
            with torch.no_grad():
                # Manual generation since our GPT model doesn't have a generate method yet
                generated = input_ids.clone()
                
                # Generate tokens one by one
                for _ in range(max_length - input_ids.size(1)):
                    with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                        outputs = model(generated)
                        next_token_logits = outputs[:, -1, :]
                        
                        # Apply temperature
                        next_token_logits = next_token_logits / temperature
                        
                        # Apply top-p (nucleus) sampling
                        if top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                            
                            # Remove tokens with cumulative probability above the threshold
                            sorted_indices_to_remove = cumulative_probs > top_p
                            # Shift the indices to the right to keep the first token above threshold
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            
                            indices_to_remove = sorted_indices[sorted_indices_to_remove]
                            next_token_logits[:, indices_to_remove] = -float('Inf')
                        
                        # Sample from the filtered distribution
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        
                        # Add the generated token to the sequence
                        generated = torch.cat((generated, next_token), dim=1)
                        
                        # Stop if we generate an EOS token
                        if next_token[0, 0].item() == tokenizer.eos_token_id:
                            break
                
                output_ids = generated
            
            # Decode
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Save result
            output_file = os.path.join(output_dir, f'sample_{i+1}.txt')
            with open(output_file, 'w') as f:
                f.write(generated_text)
            
            # Print result
            print('\\n' + '-'*50)
            print(f'Generated Text (Sample {i+1}):')
            print('-'*50)
            print(generated_text)
            print('-'*50)
            print(f'Saved to: {output_file}')
    
    except Exception as e:
        print(f'Error during text generation: {e}')

# Run generation
generate_text(
    model_path='$MODEL_PATH',
    prompt='$PROMPT',
    max_length=$MAX_LENGTH,
    temperature=$TEMPERATURE,
    top_p=$TOP_P,
    num_samples=$NUM_SAMPLES
)
"

echo "Text generation complete."