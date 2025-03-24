#!/usr/bin/env python

import os
import argparse
import torch
import logging
import json
import time
import datetime
from datasets import load_dataset
from transformers import GPT2Tokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from simple_gpt.configs import ModelConfig, TrainingConfig, DataConfig
from simple_gpt.models import GPTModel
from simple_gpt.trainers import Trainer
from simple_gpt.utils.data_utils import tokenize_function, prepare_datasets

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Also log to a file
def setup_file_logging(log_file):
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(file_handler)

class MetricsLogger:
    """Class to log metrics to a file."""
    def __init__(self, filepath):
        self.filepath = filepath
        self.metrics = []
        
    def log(self, step, metrics):
        """Log metrics for a step."""
        entry = {"step": step, "timestamp": time.time(), **metrics}
        self.metrics.append(entry)
        self._save()
        
    def _save(self):
        """Save metrics to file."""
        with open(self.filepath, "w") as f:
            json.dump(self.metrics, f, indent=2)

class CustomTrainer(Trainer):
    """Extended trainer with metrics logging."""
    def __init__(self, *args, metrics_logger=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_logger = metrics_logger
        
        # Force GPU check
        if torch.cuda.is_available():
            logger.info("CUDA is available! Moving model to GPU...")
            self.model = self.model.cuda()
            self.device = torch.device("cuda")
            
            # Print GPU usage
            logger.info(f"Initial GPU memory usage: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            logger.info(f"Initial GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        else:
            logger.warning("CUDA is not available. Training on CPU will be slow!")
        
    def train(self):
        """Train with metrics logging."""
        total_steps = 0
        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()
        
        train_iterator = range(int(self.config.num_train_epochs))
        
        # Start time for the entire training process
        training_start = time.time()
        
        for epoch in train_iterator:
            epoch_iterator = self.train_dataloader
            epoch_start = time.time()
            
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Log GPU usage periodically
                if step % 100 == 0 and torch.cuda.is_available():
                    logger.info(f"GPU memory usage: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
                    
                # Forward pass with optional mixed precision
                if self.config.fp16 and torch.cuda.is_available():
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = self._forward_pass(batch)
                        loss = outputs["loss"]
                        loss = loss / self.config.gradient_accumulation_steps
                    
                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                else:
                    outputs = self._forward_pass(batch)
                    loss = outputs["loss"]
                    loss = loss / self.config.gradient_accumulation_steps
                    loss.backward()
                
                tr_loss += loss.item()
                
                # Update model parameters after accumulating gradients
                if (step + 1) % self.config.gradient_accumulation_steps == 0 or step == len(epoch_iterator) - 1:
                    if self.config.fp16:
                        # Unscale gradients and clip them
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        
                        # Update parameters with scaled optimizer
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        
                        # Update parameters
                        self.optimizer.step()
                    
                    self.lr_scheduler.step()
                    self.model.zero_grad()
                    global_step += 1
                    
                    # Log metrics
                    if self.config.logging_steps > 0 and global_step % self.config.logging_steps == 0:
                        loss_scalar = tr_loss / global_step if global_step > 0 else 0
                        logs = {
                            "loss": loss_scalar,
                            "learning_rate": self.lr_scheduler.get_last_lr()[0],
                            "epoch": epoch + (step + 1) / len(epoch_iterator),
                            "elapsed": time.time() - training_start
                        }
                        
                        # Calculate speed in samples per second
                        samples_per_second = global_step * self.config.batch_size * self.config.gradient_accumulation_steps / (time.time() - training_start)
                        logs["samples_per_second"] = samples_per_second
                        
                        # Calculate estimated time remaining
                        if epoch > 0 or step > 0:
                            total_estimated_time = (time.time() - training_start) * self.config.num_train_epochs / (epoch + (step + 1) / len(epoch_iterator))
                            estimated_remaining = total_estimated_time - (time.time() - training_start)
                            logs["estimated_remaining"] = str(datetime.timedelta(seconds=int(estimated_remaining)))
                        
                        logger.info(f"Step {global_step}: loss={logs['loss']:.4f}, lr={logs['learning_rate']:.9f}, " +
                                   f"speed={logs['samples_per_second']:.2f} samples/s" +
                                   (f", remaining={logs['estimated_remaining']}" if 'estimated_remaining' in logs else ""))
                        
                        if self.metrics_logger:
                            self.metrics_logger.log(global_step, logs)
                    
                    # Evaluate model
                    if self.eval_dataloader is not None and self.config.eval_steps > 0 and global_step % self.config.eval_steps == 0:
                        eval_results = self.evaluate()
                        
                        # Add evaluation metrics to logs
                        eval_logs = {
                            "eval_loss": eval_results["loss"],
                            "eval_perplexity": eval_results["perplexity"],
                            "global_step": global_step,
                            "epoch": epoch + (step + 1) / len(epoch_iterator)
                        }
                        
                        logger.info(f"Evaluation at step {global_step}: " +
                                   f"loss={eval_logs['eval_loss']:.4f}, " +
                                   f"perplexity={eval_logs['eval_perplexity']:.2f}")
                        
                        if self.metrics_logger:
                            self.metrics_logger.log(global_step, eval_logs)
                    
                    # Save model checkpoint
                    if self.config.save_steps > 0 and global_step % self.config.save_steps == 0:
                        checkpoint_path = os.path.join(self.output_dir, f"checkpoint-{global_step}")
                        self.save_model(checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                
                # Check if we reached max_steps
                total_steps += 1
                if self.config.max_steps > 0 and total_steps >= self.config.max_steps:
                    # DataLoader doesn't have a close method, just break out of the loop
                    break
            
            # Log epoch completion time
            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch+1}/{self.config.num_train_epochs} completed in {str(datetime.timedelta(seconds=int(epoch_time)))}")
            
            # Check if we reached max_steps
            if self.config.max_steps > 0 and total_steps >= self.config.max_steps:
                break
            
            # Evaluate at the end of each epoch
            if self.eval_dataloader is not None:
                eval_results = self.evaluate()
                
                # Add evaluation metrics to logs
                eval_logs = {
                    "eval_loss": eval_results["loss"],
                    "eval_perplexity": eval_results["perplexity"],
                    "global_step": global_step,
                    "epoch": epoch + 1,
                    "epoch_time": epoch_time
                }
                
                logger.info(f"Evaluation at end of epoch {epoch+1}: " +
                           f"loss={eval_logs['eval_loss']:.4f}, " +
                           f"perplexity={eval_logs['eval_perplexity']:.2f}")
                
                if self.metrics_logger:
                    self.metrics_logger.log(global_step, eval_logs)
        
        # Save the final model
        self.save_model(self.output_dir)
        
        # Report total training time
        total_time = time.time() - training_start
        logger.info(f"Total training time: {str(datetime.timedelta(seconds=int(total_time)))}")
        
        # Return training metrics
        metrics = {
            "global_step": global_step, 
            "average_loss": tr_loss / global_step if global_step > 0 else 0,
            "total_training_time": total_time
        }
        
        if self.metrics_logger:
            self.metrics_logger.log(global_step, metrics)
            
        return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Run a long transformer model training session")
    
    # Model configuration
    parser.add_argument("--model_type", type=str, default="gpt", choices=["gpt", "transformer"],
                        help="Type of model to train (gpt or transformer)")
    parser.add_argument("--d_model", type=int, default=768,
                        help="Dimension of embeddings and hidden states")
    parser.add_argument("--num_heads", type=int, default=12,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=6,
                        help="Number of transformer layers")
    parser.add_argument("--d_ff", type=int, default=3072,
                        help="Dimension of feed-forward layer")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to a checkpoint to continue training from")
    
    # Training configuration
    parser.add_argument("--output_dir", type=str, default="./output/long_training",
                        help="Directory to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of steps to accumulate gradients")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max_train_time", type=int, default=12,
                        help="Maximum training time in hours (approximate)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Proportion of training to perform learning rate warmup")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--logging_steps", type=int, default=100,
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
    parser.add_argument("--use_slimpajama", action="store_true",
                        help="Use the SlimPajama-30B dataset instead of the default dataset")
    parser.add_argument("--slimpajama_subset", type=str, default="all",
                        choices=["all", "arxiv", "github", "stackexchange", "wiki", "books", "c4", "common_crawl"],
                        help="Subset of SlimPajama-30B to use")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Maximum number of training samples (None uses the entire dataset)")
    parser.add_argument("--max_val_samples", type=int, default=None,
                        help="Maximum number of validation samples (None uses the entire validation set)")
    
    # Logging
    parser.add_argument("--metrics_file", type=str, default="metrics.json",
                        help="File to save training metrics")
    parser.add_argument("--debug_mode", action="store_true",
                        help="Enable verbose debug logging")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup file logging
    log_file = os.path.join(args.output_dir, "training.log")
    setup_file_logging(log_file)
    
    # Set debug level if requested
    if args.debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug mode enabled - verbose logging activated")
    
    # Create metrics logger
    # Check if metrics_file is an absolute path or just a filename
    if os.path.isabs(args.metrics_file) or args.metrics_file.startswith('./'):
        metrics_file = args.metrics_file
    else:
        metrics_file = os.path.join(args.output_dir, args.metrics_file)
    logger.info(f"Metrics will be saved to: {metrics_file}")
    metrics_logger = MetricsLogger(metrics_file)
    
    # Log start of training and parameters
    logger.info(f"Starting long training run with parameters:")
    logger.info(f"  Model type: {args.model_type}")
    logger.info(f"  Model size: d_model={args.d_model}, num_heads={args.num_heads}, num_layers={args.num_layers}")
    logger.info(f"  Training: batch_size={args.batch_size}, grad_accum={args.gradient_accumulation_steps}, lr={args.learning_rate}")
    logger.info(f"  Duration: epochs={args.num_train_epochs}, max_time={args.max_train_time}h")
    logger.info(f"  Dataset: {args.dataset_name}/{args.dataset_config_name}, train_samples={args.max_train_samples}")
    
    # Configure HuggingFace cache - use local directory relative to output
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(args.output_dir)), "cache")
    hf_cache_dir = os.path.join(cache_dir, "huggingface")
    datasets_cache_dir = os.path.join(cache_dir, "datasets")
    
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.makedirs(datasets_cache_dir, exist_ok=True)
    
    logger.info(f"Using cache directories:")
    logger.info(f"  HuggingFace cache: {hf_cache_dir}")
    logger.info(f"  Datasets cache: {datasets_cache_dir}")
    
    # Load tokenizer (using GPT-2 tokenizer for consistency)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=hf_cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model config
    model_config = ModelConfig(
        model_type=args.model_type,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_length=args.max_seq_length,
        dropout=args.dropout
    )
    
    # If checkpoint path provided, load configuration from there
    if args.checkpoint_path:
        logger.info(f"Continuing training from checkpoint: {args.checkpoint_path}")
        model_config_path = os.path.join(args.checkpoint_path, "model_config.json")
        if os.path.exists(model_config_path):
            logger.info(f"Loading model configuration from {model_config_path}")
            with open(model_config_path, "r") as f:
                config_dict = json.load(f)
                
                # Update the model_config with the values from the saved config
                model_config.model_type = config_dict.get("model_type", model_config.model_type)
                model_config.vocab_size = config_dict.get("vocab_size", model_config.vocab_size)
                model_config.d_model = config_dict.get("d_model", model_config.d_model)
                model_config.num_heads = config_dict.get("num_heads", model_config.num_heads)
                model_config.num_layers = config_dict.get("num_layers", model_config.num_layers)
                model_config.d_ff = config_dict.get("d_ff", model_config.d_ff)
                model_config.max_seq_length = config_dict.get("max_seq_length", model_config.max_seq_length)
                model_config.dropout = config_dict.get("dropout", model_config.dropout)
                
                if "num_encoder_layers" in config_dict:
                    model_config.num_encoder_layers = config_dict["num_encoder_layers"]
                if "num_decoder_layers" in config_dict:
                    model_config.num_decoder_layers = config_dict["num_decoder_layers"]
                
            logger.info(f"Using model configuration from checkpoint: d_model={model_config.d_model}, "
                      f"num_heads={model_config.num_heads}, num_layers={model_config.num_layers}")
    
    # Calculate max_steps based on max_train_time if provided
    # Estimating 10 steps per minute for a model with d_model=768, num_layers=6, batch_size=4
    # This is a rough estimate and will vary based on hardware
    # For GPUs like RTX 4090, this could be faster
    if args.max_train_time > 0:
        estimated_steps_per_hour = 600  # 10 steps per minute * 60
        max_steps = estimated_steps_per_hour * args.max_train_time
        logger.info(f"Setting max_steps to {max_steps} based on {args.max_train_time} hours training time")
    else:
        max_steps = -1
    
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=max_steps,
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
    # Initialize model either from scratch or from checkpoint
    if args.checkpoint_path:
        try:
            model = Trainer.load_model(args.checkpoint_path, model_config)
            logger.info(f"Successfully loaded model weights from checkpoint: {args.checkpoint_path}")
        except Exception as e:
            logger.error(f"Error loading model from checkpoint: {e}")
            logger.info("Falling back to initializing a new model")
            if model_config.model_type.lower() == "gpt":
                model = GPTModel(
                    vocab_size=tokenizer.vocab_size,
                    d_model=model_config.d_model,
                    num_heads=model_config.num_heads,
                    num_layers=model_config.num_layers,
                    d_ff=model_config.d_ff,
                    max_seq_length=model_config.max_seq_length,
                    dropout=model_config.dropout
                )
            else:  # "transformer"
                model = TransformerModel(
                    vocab_size=tokenizer.vocab_size,
                    d_model=model_config.d_model,
                    num_heads=model_config.num_heads,
                    num_encoder_layers=model_config.num_encoder_layers,
                    num_decoder_layers=model_config.num_decoder_layers,
                    d_ff=model_config.d_ff,
                    max_seq_length=model_config.max_seq_length,
                    dropout=model_config.dropout
                )
    else:
        # Initialize model from scratch
        logger.info("Initializing new model from scratch")
        if model_config.model_type.lower() == "gpt":
            model = GPTModel(
                vocab_size=tokenizer.vocab_size,
                d_model=model_config.d_model,
                num_heads=model_config.num_heads,
                num_layers=model_config.num_layers,
                d_ff=model_config.d_ff,
                max_seq_length=model_config.max_seq_length,
                dropout=model_config.dropout
            )
        else:  # "transformer"
            model = TransformerModel(
                vocab_size=tokenizer.vocab_size,
                d_model=model_config.d_model,
                num_heads=model_config.num_heads,
                num_encoder_layers=model_config.num_encoder_layers,
                num_decoder_layers=model_config.num_decoder_layers,
                d_ff=model_config.d_ff,
                max_seq_length=model_config.max_seq_length,
                dropout=model_config.dropout
            )
            
    # Prepare the datasets
    if args.use_slimpajama:
        # Use the SlimPajama-30B dataset
        logger.info(f"Loading SlimPajama-30B dataset (subset: {args.slimpajama_subset})")
        
        # Configure dataset access
        # Based on dataset documentation and availability, update repository details
        if args.slimpajama_subset == "arxiv":
            slimpajama_repo = "cerebras/SlimPajama-627B-arxiv"
            slimpajama_config = None
        elif args.slimpajama_subset == "github":
            slimpajama_repo = "cerebras/SlimPajama-627B-github"
            slimpajama_config = None
        elif args.slimpajama_subset == "stackexchange":
            slimpajama_repo = "cerebras/SlimPajama-627B-stackexchange"  
            slimpajama_config = None
        elif args.slimpajama_subset == "wiki":
            slimpajama_repo = "cerebras/SlimPajama-627B-wiki"
            slimpajama_config = None
        elif args.slimpajama_subset == "books":
            slimpajama_repo = "cerebras/SlimPajama-627B-books"
            slimpajama_config = None
        elif args.slimpajama_subset == "c4":
            slimpajama_repo = "cerebras/SlimPajama-627B-c4"
            slimpajama_config = None
        elif args.slimpajama_subset == "common_crawl":
            slimpajama_repo = "cerebras/SlimPajama-627B-common_crawl"
            slimpajama_config = None
        else:
            # Use the main repository for the balanced mix of domains
            slimpajama_repo = "cerebras/SlimPajama-627B"
            slimpajama_config = "default"
            logger.info("Using SlimPajama balanced mix across all domains")
        
        logger.info(f"Attempting to load SlimPajama dataset from repository: {slimpajama_repo}")
        
        try:
            # Attempt to load the dataset with appropriate configurations
            raw_datasets = load_dataset(
                slimpajama_repo,
                slimpajama_config,
                cache_dir=datasets_cache_dir,
                trust_remote_code=True
            )
            logger.info(f"Successfully loaded SlimPajama dataset with {len(raw_datasets['train'])} training examples")
        except Exception as e:
            logger.error(f"Error loading SlimPajama dataset: {e}")
            logger.warning("Permission issues or dataset not available. Falling back to default dataset...")
            
            # Fall back to default dataset
            logger.info(f"Using fallback dataset: {data_config.dataset_name}/{data_config.dataset_config_name}")
            raw_datasets = load_dataset(
                data_config.dataset_name,
                data_config.dataset_config_name,
                cache_dir=datasets_cache_dir
            )
    else:
        # Use the configured dataset
        logger.info(f"Loading dataset: {data_config.dataset_name}/{data_config.dataset_config_name}")
        raw_datasets = load_dataset(
            data_config.dataset_name, 
            data_config.dataset_config_name,
            cache_dir=datasets_cache_dir
        )
    
    # Prepare datasets
    if args.debug_mode:
        logger.debug("Starting to prepare datasets from raw data")
        logger.debug(f"Raw dataset structure: {raw_datasets}")
        logger.debug(f"Max sequence length: {model_config.max_seq_length}")
        
        # Log dataset fields if available
        if "train" in raw_datasets:
            sample = raw_datasets["train"][0] if len(raw_datasets["train"]) > 0 else {}
            logger.debug(f"Sample data fields: {list(sample.keys())}")
    
    # For SlimPajama, ensure we use the right text field
    text_field = "text"
    if args.use_slimpajama and "train" in raw_datasets:
        # Check sample data to determine the right field name
        sample = raw_datasets["train"][0] if len(raw_datasets["train"]) > 0 else {}
        if "text" not in sample and "content" in sample:
            text_field = "content"
            logger.info(f"Using '{text_field}' as the text field for SlimPajama dataset")
    
    train_dataset, eval_dataset = prepare_datasets(
        raw_datasets, 
        tokenizer, 
        max_train_samples=data_config.max_train_samples,
        max_val_samples=data_config.max_val_samples,
        max_seq_length=model_config.max_seq_length,
        text_column_name=text_field
    )
    
    if args.debug_mode:
        logger.debug(f"Train dataset size: {len(train_dataset)} examples")
        logger.debug(f"Validation dataset size: {len(eval_dataset)} examples")
        # Print sample structure
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            logger.debug(f"Sample training example structure:")
            for key, value in sample.items():
                if hasattr(value, 'shape'):
                    logger.debug(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    logger.debug(f"  {key}: {type(value)}")
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        pad_to_multiple_of=8,
        return_tensors="pt"
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
    
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        logger.info(f"Current GPU memory usage: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
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
    trainer = CustomTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=training_config,
        output_dir=args.output_dir,
        metrics_logger=metrics_logger
    )
    
    # Save tokenizer
    tokenizer.save_pretrained(args.output_dir)
    
    # Save configuration files
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
        
    # Save model config for later loading
    model_config_dict = {
        "model_type": model_config.model_type,
        "d_model": model_config.d_model,
        "num_heads": model_config.num_heads,
        "num_layers": model_config.num_layers,
        "d_ff": model_config.d_ff,
        "max_seq_length": model_config.max_seq_length,
        "dropout": model_config.dropout,
        "vocab_size": len(tokenizer)
    }
    
    with open(os.path.join(args.output_dir, "model_config.json"), "w") as f:
        json.dump(model_config_dict, f, indent=2)
    
    # Train model
    logger.info("Starting training")
    if args.debug_mode:
        logger.debug("Debug information before training:")
        logger.debug(f"Model architecture summary: {model}")
        logger.debug(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.debug(f"Training config: {vars(training_config)}")
        logger.debug(f"First batch check:")
        
        # Try getting a sample batch
        first_batch = next(iter(train_dataloader))
        for key, value in first_batch.items():
            logger.debug(f"  {key}: shape={value.shape if hasattr(value, 'shape') else 'N/A'}")
            
        logger.debug("Attempting a forward pass with sample batch...")
    
    train_start = time.time()
    trainer.train()
    train_end = time.time()
    
    # Final evaluation
    logger.info("Performing final evaluation")
    eval_results = trainer.evaluate()
    logger.info(f"Final evaluation results: {eval_results}")
    
    # Final save
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    
    # Calculate and log total training time
    total_training_time = train_end - train_start
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Training complete!")
    logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Add a summary file
    with open(os.path.join(args.output_dir, "training_summary.txt"), "w") as f:
        f.write(f"Training Summary\n")
        f.write(f"===============\n\n")
        f.write(f"Model Configuration:\n")
        f.write(f"  Type: {args.model_type}\n")
        f.write(f"  Embedding dimension: {args.d_model}\n")
        f.write(f"  Attention heads: {args.num_heads}\n")
        f.write(f"  Layers: {args.num_layers}\n")
        f.write(f"  Feed-forward dimension: {args.d_ff}\n\n")
        
        f.write(f"Training Configuration:\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}\n")
        f.write(f"  Learning rate: {args.learning_rate}\n")
        f.write(f"  Training epochs: {args.num_train_epochs}\n")
        f.write(f"  Training samples: {args.max_train_samples}\n\n")
        
        f.write(f"Training Results:\n")
        f.write(f"  Final loss: {eval_results['loss']:.4f}\n")
        f.write(f"  Final perplexity: {eval_results['perplexity']:.2f}\n")
        f.write(f"  Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s\n")


if __name__ == "__main__":
    main()