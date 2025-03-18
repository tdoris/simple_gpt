#!/usr/bin/env python

import os
import argparse
import json
import logging
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from simple_gpt.configs import ModelConfig, TrainingConfig, DataConfig
from simple_gpt.models import GPTModel, TransformerModel
from simple_gpt.trainers import Trainer
from simple_gpt.utils.data_utils import prepare_datasets

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Continue training from a checkpoint")
    
    # Model checkpoint
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model checkpoint")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save model checkpoints (defaults to model_path + '_continued')")
    
    # Training configuration overrides
    parser.add_argument("--num_train_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Learning rate")
    
    # Data configuration overrides
    parser.add_argument("--max_train_samples", type=int, default=10000,
                        help="Maximum number of training samples")
    parser.add_argument("--max_val_samples", type=int, default=1000,
                        help="Maximum number of validation samples")
    
    return parser.parse_args()


def load_config(model_path, config_file):
    """Load a configuration file from the model path."""
    config_path = os.path.join(model_path, config_file)
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return None


def main():
    args = parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.model_path + "_continued"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model configuration
    model_config_dict = load_config(args.model_path, "model_config.json")
    if model_config_dict is None:
        logger.warning("No model_config.json found. Using default configuration.")
        model_config = ModelConfig()
    else:
        logger.info(f"Loaded model configuration from {args.model_path}/model_config.json")
        model_config = ModelConfig(
            model_type=model_config_dict.get("model_type", "gpt"),
            vocab_size=model_config_dict.get("vocab_size", 50257),
            d_model=model_config_dict.get("d_model", 768),
            num_heads=model_config_dict.get("num_heads", 12),
            num_layers=model_config_dict.get("num_layers", 12),
            d_ff=model_config_dict.get("d_ff", 3072),
            max_seq_length=model_config_dict.get("max_seq_length", 1024),
            dropout=model_config_dict.get("dropout", 0.1)
        )
    
    # Load training configuration
    training_config_dict = load_config(args.model_path, "training_config.json")
    if training_config_dict is None:
        logger.warning("No training_config.json found. Using default configuration.")
        training_config = TrainingConfig()
    else:
        logger.info(f"Loaded training configuration from {args.model_path}/training_config.json")
        training_config = TrainingConfig(**training_config_dict)
    
    # Override training config with command line arguments
    if args.num_train_epochs is not None:
        training_config.num_train_epochs = args.num_train_epochs
    if args.batch_size is not None:
        training_config.batch_size = args.batch_size
    if args.learning_rate is not None:
        training_config.learning_rate = args.learning_rate
    
    # Load tokenizer
    tokenizer_path = args.model_path
    if not os.path.exists(os.path.join(args.model_path, "vocab.json")):
        logger.info("No tokenizer found in model path. Using GPT-2 tokenizer.")
        tokenizer_path = "gpt2"
    
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = Trainer.load_model(args.model_path, model_config)
    logger.info(f"Loaded model from {args.model_path}")
    
    # Prepare datasets
    logger.info("Loading datasets")
    raw_datasets = load_dataset("wikitext", "wikitext-103-raw-v1")
    
    train_dataset, eval_dataset = prepare_datasets(
        raw_datasets, 
        tokenizer, 
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
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
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=training_config,
        output_dir=args.output_dir
    )
    
    # Train model
    logger.info(f"Starting continued training for {training_config.num_train_epochs} epochs")
    trainer.train()
    
    # Final evaluation
    logger.info("Performing final evaluation")
    eval_results = trainer.evaluate()
    logger.info(f"Final evaluation results: {eval_results}")
    
    # Save model and tokenizer
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save evaluation results
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=2)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()