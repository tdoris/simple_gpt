#!/usr/bin/env python

import os
import argparse
import logging
import torch
from transformers import GPT2Tokenizer
from simple_gpt.configs import ModelConfig, TrainingConfig, DataConfig
from simple_gpt.models import GPTModel, TransformerModel
from simple_gpt.utils.data_utils import prepare_dataset, get_dataloader
from simple_gpt.trainers import Trainer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPT-style language model")
    
    # Model configurations
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model_type", type=str, default="gpt", choices=["gpt", "transformer"],
                            help="Type of model to train (gpt or transformer)")
    model_group.add_argument("--vocab_size", type=int, default=50257,
                            help="Vocabulary size")
    model_group.add_argument("--d_model", type=int, default=768,
                            help="Dimension of embeddings and hidden states")
    model_group.add_argument("--num_heads", type=int, default=12,
                            help="Number of attention heads")
    model_group.add_argument("--num_layers", type=int, default=12,
                            help="Number of transformer layers")
    model_group.add_argument("--d_ff", type=int, default=3072,
                            help="Dimension of feed-forward layer")
    model_group.add_argument("--max_seq_length", type=int, default=1024,
                            help="Maximum sequence length")
    model_group.add_argument("--dropout", type=float, default=0.1,
                            help="Dropout rate")
    
    # Training configurations
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument("--batch_size", type=int, default=8,
                           help="Batch size for training")
    train_group.add_argument("--gradient_accumulation_steps", type=int, default=1,
                           help="Number of updates steps to accumulate before backward pass")
    train_group.add_argument("--learning_rate", type=float, default=5e-5,
                           help="Learning rate")
    train_group.add_argument("--weight_decay", type=float, default=0.01,
                           help="Weight decay")
    train_group.add_argument("--num_train_epochs", type=int, default=3,
                           help="Number of training epochs")
    train_group.add_argument("--max_steps", type=int, default=-1,
                           help="Number of training steps (overrides num_train_epochs)")
    train_group.add_argument("--warmup_ratio", type=float, default=0.1,
                           help="Ratio of warmup steps")
    train_group.add_argument("--fp16", action="store_true",
                           help="Use mixed precision training")
    train_group.add_argument("--logging_steps", type=int, default=100,
                           help="Logging steps")
    train_group.add_argument("--eval_steps", type=int, default=500,
                           help="Evaluation steps")
    train_group.add_argument("--save_steps", type=int, default=1000,
                           help="Save steps")
    train_group.add_argument("--save_total_limit", type=int, default=2,
                           help="Limit the number of checkpoints saved")
    train_group.add_argument("--seed", type=int, default=42,
                           help="Random seed")
    
    # Data configurations
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument("--dataset_name", type=str, default="wikitext",
                          help="Dataset name from Hugging Face Datasets")
    data_group.add_argument("--dataset_config_name", type=str, default="wikitext-103-raw-v1",
                          help="Dataset configuration name")
    data_group.add_argument("--train_file", type=str, default=None,
                          help="Path to local training file")
    data_group.add_argument("--validation_file", type=str, default=None,
                          help="Path to local validation file")
    data_group.add_argument("--max_train_samples", type=int, default=None,
                          help="Maximum number of training samples")
    data_group.add_argument("--max_val_samples", type=int, default=None,
                          help="Maximum number of validation samples")
    data_group.add_argument("--preprocessing_num_workers", type=int, default=4,
                          help="Number of workers for preprocessing")
    data_group.add_argument("--overwrite_cache", action="store_true",
                          help="Overwrite cached files")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./output",
                      help="Directory to store output files")
    parser.add_argument("--use_wandb", action="store_true",
                      help="Use Weights & Biases for logging")
    
    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Create configurations
    model_config = ModelConfig(
        model_type=args.model_type,
        vocab_size=args.vocab_size,
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
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed
    )
    
    data_config = DataConfig(
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        train_file=args.train_file,
        validation_file=args.validation_file,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        preprocessing_num_workers=args.preprocessing_num_workers,
        overwrite_cache=args.overwrite_cache
    )
    
    logger.info(f"Model configuration: {model_config}")
    logger.info(f"Training configuration: {training_config}")
    logger.info(f"Data configuration: {data_config}")
    
    # Load tokenizer (using Hugging Face's GPT2Tokenizer)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    train_dataset, eval_dataset = prepare_dataset(data_config, tokenizer)
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
    
    # Create dataloaders
    train_dataloader = get_dataloader(
        train_dataset, 
        batch_size=training_config.batch_size, 
        shuffle=True, 
        num_workers=data_config.preprocessing_num_workers
    )
    
    eval_dataloader = None
    if eval_dataset:
        eval_dataloader = get_dataloader(
            eval_dataset, 
            batch_size=training_config.batch_size, 
            shuffle=False, 
            num_workers=data_config.preprocessing_num_workers
        )
    
    # Create model
    if model_config.model_type.lower() == "gpt":
        model = GPTModel(
            vocab_size=model_config.vocab_size,
            d_model=model_config.d_model,
            num_heads=model_config.num_heads,
            num_layers=model_config.num_layers,
            d_ff=model_config.d_ff,
            max_seq_length=model_config.max_seq_length,
            dropout=model_config.dropout
        )
    else:  # "transformer"
        model = TransformerModel(
            vocab_size=model_config.vocab_size,
            d_model=model_config.d_model,
            num_heads=model_config.num_heads,
            num_encoder_layers=model_config.num_encoder_layers,
            num_decoder_layers=model_config.num_decoder_layers,
            d_ff=model_config.d_ff,
            max_seq_length=model_config.max_seq_length,
            dropout=model_config.dropout
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=training_config,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb
    )
    
    # Train model
    logger.info("Starting training...")
    metrics = trainer.train()
    logger.info(f"Training completed with metrics: {metrics}")
    
    # Save tokenizer
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info(f"Model and tokenizer saved to {args.output_dir}")


if __name__ == "__main__":
    main()
