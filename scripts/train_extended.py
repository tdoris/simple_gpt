#!/usr/bin/env python

import os
import argparse
import torch
import logging
from datasets import load_dataset
from transformers import GPT2Tokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from simple_gpt.configs import ModelConfig, TrainingConfig, DataConfig
from simple_gpt.models import GPTModel
from simple_gpt.trainers import Trainer
from simple_gpt.utils.data_utils import tokenize_function, prepare_datasets

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a transformer language model")
    
    # Model configuration
    parser.add_argument("--model_type", type=str, default="gpt", choices=["gpt", "transformer"],
                        help="Type of model to train (gpt or transformer)")
    parser.add_argument("--d_model", type=int, default=512,
                        help="Dimension of embeddings and hidden states")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=6,
                        help="Number of transformer layers")
    parser.add_argument("--d_ff", type=int, default=2048,
                        help="Dimension of feed-forward layer")
    parser.add_argument("--max_seq_length", type=int, default=1024,
                        help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    
    # Training configuration
    parser.add_argument("--output_dir", type=str, default="./output/long_training",
                        help="Directory to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Number of steps to accumulate gradients")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Proportion of training to perform learning rate warmup")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Logging frequency in steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluation frequency in steps")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Checkpoint saving frequency in steps")
    
    # Data configuration
    parser.add_argument("--dataset_name", type=str, default="wikitext",
                        help="Dataset name from Hugging Face datasets")
    parser.add_argument("--dataset_config_name", type=str, default="wikitext-103-raw-v1",
                        help="Dataset configuration name")
    parser.add_argument("--max_train_samples", type=int, default=10000,
                        help="Maximum number of training samples")
    parser.add_argument("--max_val_samples", type=int, default=1000,
                        help="Maximum number of validation samples")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer (using GPT-2 tokenizer for consistency)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create configs
    model_config = ModelConfig(
        model_type=args.model_type,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_length=args.max_seq_length,
        dropout=args.dropout
    )
    
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps
    )
    
    data_config = DataConfig(
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples
    )
    
    # Load and prepare datasets
    logger.info(f"Loading dataset: {data_config.dataset_name}/{data_config.dataset_config_name}")
    raw_datasets = load_dataset(data_config.dataset_name, data_config.dataset_config_name)
    
    # Prepare datasets
    train_dataset, eval_dataset = prepare_datasets(
        raw_datasets, 
        tokenizer, 
        max_train_samples=data_config.max_train_samples,
        max_val_samples=data_config.max_val_samples,
        max_seq_length=model_config.max_seq_length
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        collate_fn=data_collator,
        shuffle=True
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=training_config.batch_size,
        collate_fn=data_collator
    )
    
    # Initialize model
    logger.info(f"Initializing {model_config.model_type} model")
    if model_config.model_type.lower() == "gpt":
        model = GPTModel(
            vocab_size=len(tokenizer),
            d_model=model_config.d_model,
            num_heads=model_config.num_heads,
            num_layers=model_config.num_layers,
            d_ff=model_config.d_ff,
            max_seq_length=model_config.max_seq_length,
            dropout=model_config.dropout
        )
    else:
        from simple_gpt.models import TransformerModel
        model = TransformerModel(
            vocab_size=len(tokenizer),
            d_model=model_config.d_model,
            num_heads=model_config.num_heads,
            num_encoder_layers=model_config.num_encoder_layers,
            num_decoder_layers=model_config.num_decoder_layers,
            d_ff=model_config.d_ff,
            max_seq_length=model_config.max_seq_length,
            dropout=model_config.dropout
        )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=training_config,
        output_dir=args.output_dir
    )
    
    # Save tokenizer
    tokenizer.save_pretrained(args.output_dir)
    
    # Train model
    logger.info("Starting training")
    trainer.train()
    
    # Final evaluation
    logger.info("Performing final evaluation")
    eval_results = trainer.evaluate()
    logger.info(f"Final evaluation results: {eval_results}")
    
    # Final save
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()