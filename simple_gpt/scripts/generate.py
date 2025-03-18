#!/usr/bin/env python

import os
import argparse
import torch
import logging
from transformers import GPT2Tokenizer
from simple_gpt.configs import ModelConfig, GenerationConfig
from simple_gpt.trainers import Trainer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text using a trained language model")
    
    # Model path
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model directory")
    
    # Model configuration
    parser.add_argument("--model_type", type=str, default="gpt", choices=["gpt", "transformer"],
                        help="Type of model to use (gpt or transformer)")
    parser.add_argument("--vocab_size", type=int, default=50257,
                        help="Vocabulary size")
    parser.add_argument("--d_model", type=int, default=768,
                        help="Dimension of embeddings and hidden states")
    parser.add_argument("--num_heads", type=int, default=12,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=12,
                        help="Number of transformer layers")
    parser.add_argument("--d_ff", type=int, default=3072,
                        help="Dimension of feed-forward layer")
    parser.add_argument("--max_seq_length", type=int, default=1024,
                        help="Maximum sequence length")
    
    # Generation configuration
    parser.add_argument("--prompt", type=str, required=True,
                        help="Prompt for text generation")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum length of generated text (including prompt)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k filtering parameter")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p (nucleus) filtering parameter")
    parser.add_argument("--do_sample", action="store_true",
                        help="Sample from the distribution instead of greedy decoding")
    parser.add_argument("--num_return_sequences", type=int, default=1,
                        help="Number of sequences to generate")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(
        args.model_path if os.path.exists(os.path.join(args.model_path, "vocab.json")) else "gpt2"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model configuration
    model_config = ModelConfig(
        model_type=args.model_type,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_length=args.max_seq_length,
    )
    
    # Load model
    model = Trainer.load_model(args.model_path, model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {args.model_path}")
    
    # Process input prompt
    encoded_prompt = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    
    logger.info(f"Generating text with prompt: {args.prompt}")
    
    # Generate text
    with torch.no_grad():
        # For GPT model, we use the built-in generate method
        output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_new_tokens=args.max_length - encoded_prompt.size(1),
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=args.do_sample,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode and print generated text
    for i, sequence in enumerate(output_sequences):
        generated_text = tokenizer.decode(sequence, skip_special_tokens=True)
        print(f"\nGenerated sequence {i+1}:\n{generated_text}")


if __name__ == "__main__":
    main()
