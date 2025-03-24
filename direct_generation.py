#!/usr/bin/env python

import os
import sys
import json
import torch
import logging
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Direct text generation using HuggingFace GPT-2")
    
    parser.add_argument("--model_size", type=str, default="small", 
                        choices=["small", "medium", "large", "xl"],
                        help="GPT-2 model size to use")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Prompt for text generation")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum length of generated text (including prompt)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling (higher = more random)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k filtering parameter (0 to disable)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p filtering parameter (1.0 to disable)")
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                        help="Repetition penalty (1.0 = no penalty, >1.0 decreases repetition)")
    parser.add_argument("--do_sample", action="store_true", default=True,
                        help="Sample from the distribution instead of greedy decoding")
    parser.add_argument("--num_return_sequences", type=int, default=1,
                        help="Number of sequences to generate")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Map model size to HuggingFace model name
    model_map = {
        "small": "gpt2",
        "medium": "gpt2-medium",
        "large": "gpt2-large",
        "xl": "gpt2-xl"
    }
    model_name = model_map[args.model_size]
    
    # Set cache directory
    cache_dir = "./output/cache/huggingface"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_dir, "transformers")
    
    logger.info(f"Loading {model_name} from HuggingFace directly...")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the model
    model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    logger.info(f"Using device: {device}")
    logger.info(f"Generating text with prompt: {args.prompt}")
    
    # Encode prompt with attention mask
    inputs = tokenizer(args.prompt, return_tensors='pt', padding=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Set generation parameters
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.do_sample,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=args.num_return_sequences
    )
    
    # Decode and print each generated sequence
    for i, sequence in enumerate(output):
        generated_text = tokenizer.decode(sequence, skip_special_tokens=True)
        print(f"\nGenerated sequence {i+1}:\n{generated_text}")

if __name__ == "__main__":
    main()