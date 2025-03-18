import os
import math
import torch
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from tqdm import tqdm
from simple_gpt.configs import TrainingConfig, ModelConfig
from simple_gpt.models import GPTModel, TransformerModel
from transformers import get_scheduler

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for training language models."""
    
    def __init__(
        self,
        model: Union[GPTModel, TransformerModel],
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        config: TrainingConfig = TrainingConfig(),
        output_dir: str = "./output",
        use_wandb: bool = False
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config
        self.output_dir = output_dir
        
        # Set random seed for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Setup optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_params = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_params,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon
        )
        
        # Setup learning rate scheduler
        num_update_steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
        if config.max_steps > 0:
            max_steps = config.max_steps
        else:
            max_steps = config.num_train_epochs * num_update_steps_per_epoch
        
        self.lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=int(max_steps * config.warmup_ratio),
            num_training_steps=max_steps
        )
        
        # Setup mixed precision training if requested
        self.scaler = torch.cuda.amp.GradScaler() if config.fp16 else None
        
        # Setup wandb
        self.use_wandb = use_wandb and has_wandb
        if self.use_wandb:
            wandb.init(project="simple-gpt", config=vars(config))
            wandb.watch(model)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def train(self) -> Dict[str, float]:
        """Train the model."""
        total_steps = 0
        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()
        
        train_iterator = range(int(self.config.num_train_epochs))
        
        for epoch in train_iterator:
            epoch_iterator = tqdm(
                self.train_dataloader, 
                desc=f"Epoch {epoch+1}/{self.config.num_train_epochs}"
            )
            
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass with optional mixed precision
                if self.config.fp16:
                    with torch.cuda.amp.autocast():
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
                        logs = {}
                        loss_scalar = tr_loss / global_step if global_step > 0 else 0
                        logs["loss"] = loss_scalar
                        logs["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
                        logger.info(f"Step {global_step}: {logs}")
                        
                        if self.use_wandb:
                            wandb.log({"train/" + k: v for k, v in logs.items()}, step=global_step)
                    
                    # Evaluate model
                    if self.eval_dataloader is not None and self.config.eval_steps > 0 and global_step % self.config.eval_steps == 0:
                        eval_results = self.evaluate()
                        logger.info(f"Evaluation results at step {global_step}: {eval_results}")
                        
                        if self.use_wandb:
                            wandb.log({"eval/" + k: v for k, v in eval_results.items()}, step=global_step)
                    
                    # Save model checkpoint
                    if self.config.save_steps > 0 and global_step % self.config.save_steps == 0:
                        self.save_model(os.path.join(self.output_dir, f"checkpoint-{global_step}"))
                
                # Update description with current loss
                epoch_iterator.set_description(
                    f"Epoch {epoch+1}/{self.config.num_train_epochs} (loss: {tr_loss/global_step:.4f})"
                )
                
                # Check if we reached max_steps
                total_steps += 1
                if self.config.max_steps > 0 and total_steps >= self.config.max_steps:
                    epoch_iterator.close()
                    break
            
            # Check if we reached max_steps
            if self.config.max_steps > 0 and total_steps >= self.config.max_steps:
                break
            
            # Evaluate at the end of each epoch
            if self.eval_dataloader is not None:
                eval_results = self.evaluate()
                logger.info(f"Evaluation results at end of epoch {epoch+1}: {eval_results}")
                
                if self.use_wandb:
                    wandb.log({"eval/" + k: v for k, v in eval_results.items()}, step=global_step)
        
        # Save the final model
        self.save_model(self.output_dir)
        
        # Return training metrics
        metrics = {"global_step": global_step, "average_loss": tr_loss / global_step if global_step > 0 else 0}
        return metrics
    
    def _forward_pass(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass and calculate loss."""
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        labels = batch.get("labels", None)
        
        # For GPT model (decoder-only transformer)
        if isinstance(self.model, GPTModel):
            logits = self.model(input_ids, attention_mask)
            
            # Calculate loss
            if labels is not None:
                # Shift logits and labels for next-token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Calculate cross-entropy loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            else:
                loss = None
            
            return {"logits": logits, "loss": loss}
        
        # For full encoder-decoder transformer
        elif isinstance(self.model, TransformerModel):
            # Assuming labels are the target sequence with a shift of 1
            tgt_input = input_ids[..., :-1]
            tgt_mask = self._generate_square_subsequent_mask(tgt_input.size(1)).to(self.device)
            
            # Create source mask if attention_mask is provided
            src_mask = None
            if attention_mask is not None:
                src_mask = attention_mask.to(self.device)
            
            # Forward pass through the model
            output = self.model(input_ids, tgt_input, src_mask, tgt_mask)
            
            # Calculate loss
            if labels is not None:
                # Flatten the output and labels
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(output.view(-1, output.size(-1)), labels.view(-1))
            else:
                loss = None
            
            return {"logits": output, "loss": loss}
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate a square mask for the sequence."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on the evaluation dataset."""
        if self.eval_dataloader is None:
            logger.warning("No evaluation dataloader provided. Skipping evaluation.")
            return {}
        
        self.model.eval()
        eval_loss = 0.0
        eval_steps = 0
        
        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with torch.no_grad():
                outputs = self._forward_pass(batch)
                loss = outputs["loss"]
                eval_loss += loss.item()
            
            eval_steps += 1
        
        # Calculate perplexity
        eval_loss = eval_loss / eval_steps if eval_steps > 0 else 0
        perplexity = math.exp(eval_loss) if eval_loss < 30 else float('inf')
        
        metrics = {"loss": eval_loss, "perplexity": perplexity}
        return metrics
    
    def save_model(self, output_dir: str):
        """Save the model, tokenizer, and training arguments."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model state_dict
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        
        # Save training config
        config_dict = vars(self.config)
        with open(os.path.join(output_dir, "training_config.json"), "w") as f:
            import json
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Model saved to {output_dir}")
    
    @classmethod
    def load_model(cls, model_path: str, model_config: ModelConfig):
        """Load a saved model."""
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
        
        # Load state dict
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state_dict)
        
        return model