#!/usr/bin/env python

import os
import sys
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
                        help="Temperature for sampling (higher = more random)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k filtering parameter (0 to disable)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p (nucleus) filtering parameter (1.0 to disable)")
    parser.add_argument("--do_sample", action="store_true",
                        help="Sample from the distribution instead of greedy decoding")
    parser.add_argument("--num_return_sequences", type=int, default=1,
                        help="Number of sequences to generate")
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                        help="Repetition penalty (1.0 = no penalty, >1.0 decreases repetition)")
    parser.add_argument("--min_length", type=int, default=0,
                        help="Minimum length of the sequence to be generated")
    # No longer needed as we automatically use HuggingFace models for GPT-2 models
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(
        args.model_path if os.path.exists(os.path.join(args.model_path, "vocab.json")) else "gpt2"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if this is a GPT-2 model path
    is_gpt2_model = "gpt2" in args.model_path.lower()
    if is_gpt2_model:
        logger.info("Detected GPT-2 model, using appropriate settings")
        
        # For GPT-2 models, try to use the HuggingFace model directly
        # This is the most reliable method for GPT-2 models
        try:
            from transformers import GPT2LMHeadModel
            logger.info("Trying to load HuggingFace GPT-2 model directly...")
            
            # Map model sizes to HuggingFace model names
            if "small" in args.model_path.lower():
                hf_model_name = "gpt2"
            elif "medium" in args.model_path.lower():
                hf_model_name = "gpt2-medium"
            elif "large" in args.model_path.lower():
                hf_model_name = "gpt2-large"
            elif "xl" in args.model_path.lower():
                hf_model_name = "gpt2-xl"
            else:
                hf_model_name = "gpt2"  # Default to small
            
            logger.info(f"Using HuggingFace {hf_model_name} model directly")
            
            # Always use HuggingFace model for GPT-2 to ensure correct generation
            # This avoids the problem with our custom models and GPT-2 weights
            hf_model = GPT2LMHeadModel.from_pretrained(hf_model_name)
            hf_model.to(device)
            hf_model.eval()
            
            # Process input prompt
            inputs = tokenizer(args.prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate text with the HuggingFace model
            logger.info(f"Generating text with prompt: {args.prompt}")
            with torch.no_grad():
                output_sequences = hf_model.generate(
                    **inputs,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    do_sample=args.do_sample,
                    num_return_sequences=args.num_return_sequences,
                    repetition_penalty=args.repetition_penalty,
                    min_length=args.min_length + inputs["input_ids"].size(1),
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            # Decode and print the generated text
            for i, sequence in enumerate(output_sequences):
                generated_text = tokenizer.decode(sequence, skip_special_tokens=True)
                print(f"\nGenerated sequence {i+1}:\n{generated_text}")
            
            # Exit early since we've already generated text using HuggingFace
            return
        except Exception as e:
            logger.warning(f"Could not use HuggingFace model directly: {e}")
            logger.info("Falling back to custom model implementation")
    
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
        # For GPT model, we use the built-in generate method with enhanced parameters
        output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_new_tokens=args.max_length - encoded_prompt.size(1),
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=args.do_sample,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=args.repetition_penalty,
            min_length=args.min_length
        )
    
    # Decode and print generated text
    for i, sequence in enumerate(output_sequences):
        # First try decoding with skip_special_tokens=True
        generated_text = tokenizer.decode(sequence, skip_special_tokens=True)
        
        # Clean up potential unicode issues for display
        cleaned_text = ""
        for c in generated_text:
            if ord(c) < 127 or c.isalpha() or c.isdigit() or c.isspace() or c in '.,!?;:-_()[]{}"\'':
                cleaned_text += c
            else:
                # Replace unknown characters with a placeholder
                cleaned_text += "□"
        
        print(f"\nGenerated sequence {i+1}:\n{generated_text}")
        
        # Debug tokenization if there are issues
        if any(c for c in generated_text if ord(c) > 127):
            print("\nDebug token information:")
            tokens = tokenizer.convert_ids_to_tokens(sequence)
            for j, token in enumerate(tokens[:20]):  # Show first 20 tokens
                print(f"Token {j}: {token} -> {tokenizer.decode([sequence[j]])}")
            
            # For debugging unicode issues
            if i == 0:  # Only for the first sequence
                import sys
                print("\nNote: Some unicode characters may not display properly in your terminal.")
                print("Using cleaned text for better readability:")
                print(cleaned_text)


if __name__ == "__main__":
    main()
