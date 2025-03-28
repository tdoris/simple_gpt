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
        self.scaler = torch.amp.GradScaler('cuda') if config.fp16 else None
        
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
                if global_step > 0:
                    current_loss = tr_loss / global_step
                else:
                    current_loss = 0.0
                    
                epoch_iterator.set_description(
                    f"Epoch {epoch+1}/{self.config.num_train_epochs} (loss: {current_loss:.4f})"
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
            
        # Save model architecture config
        model_config = {}
        if isinstance(self.model, GPTModel):
            model_config = {
                "model_type": "gpt",
                "vocab_size": self.model.token_embedding.num_embeddings,
                "d_model": self.model.d_model,
                "num_heads": self.model.layers[0].self_attention.num_heads if self.model.layers else 0,
                "num_layers": len(self.model.layers),
                "d_ff": self.model.layers[0].feed_forward.fc1.out_features if self.model.layers else 0,
                "max_seq_length": self.model.max_seq_length,
                "dropout": self.model.dropout.p
            }
        elif isinstance(self.model, TransformerModel):
            model_config = {
                "model_type": "transformer",
                "vocab_size": self.model.embedding.num_embeddings,
                "d_model": self.model.d_model,
                "num_heads": self.model.encoder_layers[0].self_attention.num_heads if self.model.encoder_layers else 0,
                "num_layers": max(len(self.model.encoder_layers), len(self.model.decoder_layers)),
                "num_encoder_layers": len(self.model.encoder_layers),
                "num_decoder_layers": len(self.model.decoder_layers),
                "d_ff": self.model.encoder_layers[0].feed_forward.fc1.out_features if self.model.encoder_layers else 0,
                "max_seq_length": self.model.positional_encoding.pe.size(1),
                "dropout": self.model.dropout.p
            }
            
        with open(os.path.join(output_dir, "model_config.json"), "w") as f:
            import json
            json.dump(model_config, f, indent=2)
        
        logger.info(f"Model saved to {output_dir}")
    
    @classmethod
    def load_model(cls, model_path: str, model_config: ModelConfig):
        """Load a saved model."""
        # Check if there's a saved model_config.json file and use it instead of the provided config
        model_config_path = os.path.join(model_path, "model_config.json")
        if os.path.exists(model_config_path):
            logger.info(f"Loading model configuration from {model_config_path}")
            with open(model_config_path, "r") as f:
                import json
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
                
            logger.info(f"Using model configuration: d_model={model_config.d_model}, "
                      f"num_heads={model_config.num_heads}, num_layers={model_config.num_layers}")
        
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
        
        # Check if this is a HuggingFace model format or our format
        # HF models have keys like 'transformer.h.0.attn.c_attn.weight'
        is_hf_model = any('transformer.h' in key for key in state_dict.keys())
        
        if is_hf_model:
            logger.info("Detected HuggingFace GPT-2 model format, converting weights...")
            # Convert HuggingFace GPT-2 weights to our format
            converted_state_dict = {}
            d_model = model_config.d_model
            num_layers = model_config.num_layers
            
            # Map the embeddings and final layer norm
            if 'transformer.wte.weight' in state_dict:
                converted_state_dict["token_embedding.weight"] = state_dict["transformer.wte.weight"]
                converted_state_dict["lm_head.weight"] = state_dict["transformer.wte.weight"]  # Weight tying
            if 'transformer.wpe.weight' in state_dict:
                converted_state_dict["position_embedding.weight"] = state_dict["transformer.wpe.weight"]
            if 'transformer.ln_f.weight' in state_dict:
                converted_state_dict["norm.weight"] = state_dict["transformer.ln_f.weight"]
            if 'transformer.ln_f.bias' in state_dict:
                converted_state_dict["norm.bias"] = state_dict["transformer.ln_f.bias"]
            
            # Map each layer
            for i in range(num_layers):
                hf_prefix = f"transformer.h.{i}."
                our_prefix = f"layers.{i}."
                
                # Layer normalization 1
                if hf_prefix + "ln_1.weight" in state_dict:
                    converted_state_dict[our_prefix + "norm1.weight"] = state_dict[hf_prefix + "ln_1.weight"]
                if hf_prefix + "ln_1.bias" in state_dict:
                    converted_state_dict[our_prefix + "norm1.bias"] = state_dict[hf_prefix + "ln_1.bias"]
                
                # Layer normalization 2
                if hf_prefix + "ln_2.weight" in state_dict:
                    converted_state_dict[our_prefix + "norm2.weight"] = state_dict[hf_prefix + "ln_2.weight"]
                if hf_prefix + "ln_2.bias" in state_dict:
                    converted_state_dict[our_prefix + "norm2.bias"] = state_dict[hf_prefix + "ln_2.bias"]
                
                # Self-attention weights - GPT-2 uses a combined QKV projection
                # We need to split it for our model
                if hf_prefix + "attn.c_attn.weight" in state_dict:
                    c_attn_weight = state_dict[hf_prefix + "attn.c_attn.weight"]
                    c_attn_bias = state_dict[hf_prefix + "attn.c_attn.bias"]
                    
                    # Split into query, key, value sections
                    split_size = d_model
                    w_splits = torch.split(c_attn_weight, split_size, dim=1)
                    b_splits = torch.split(c_attn_bias, split_size)
                    
                    qw, kw, vw = w_splits
                    qb, kb, vb = b_splits
                    
                    # Transpose to match our model's format
                    qw = qw.transpose(0, 1)
                    kw = kw.transpose(0, 1)
                    vw = vw.transpose(0, 1)
                    
                    # Store query, key, value weights and biases
                    converted_state_dict[our_prefix + "self_attention.query.weight"] = qw
                    converted_state_dict[our_prefix + "self_attention.query.bias"] = qb
                    converted_state_dict[our_prefix + "self_attention.key.weight"] = kw
                    converted_state_dict[our_prefix + "self_attention.key.bias"] = kb
                    converted_state_dict[our_prefix + "self_attention.value.weight"] = vw
                    converted_state_dict[our_prefix + "self_attention.value.bias"] = vb
                
                # Output projection
                if hf_prefix + "attn.c_proj.weight" in state_dict:
                    c_proj_weight = state_dict[hf_prefix + "attn.c_proj.weight"]
                    c_proj_bias = state_dict[hf_prefix + "attn.c_proj.bias"]
                    
                    # Transpose the projection weight to match our model's expected shape
                    converted_state_dict[our_prefix + "self_attention.output_layer.weight"] = c_proj_weight.transpose(0, 1)
                    converted_state_dict[our_prefix + "self_attention.output_layer.bias"] = c_proj_bias
                
                # Feed forward network
                if hf_prefix + "mlp.c_fc.weight" in state_dict:
                    # Transpose to match our model's expectations
                    converted_state_dict[our_prefix + "feed_forward.fc1.weight"] = state_dict[hf_prefix + "mlp.c_fc.weight"].transpose(0, 1)
                    converted_state_dict[our_prefix + "feed_forward.fc1.bias"] = state_dict[hf_prefix + "mlp.c_fc.bias"]
                    converted_state_dict[our_prefix + "feed_forward.fc2.weight"] = state_dict[hf_prefix + "mlp.c_proj.weight"].transpose(0, 1)
                    converted_state_dict[our_prefix + "feed_forward.fc2.bias"] = state_dict[hf_prefix + "mlp.c_proj.bias"]
            
            # Use the converted state dict
            state_dict = converted_state_dict
            logger.info("Successfully converted HuggingFace GPT-2 weights to our format")
        
        # Try to load the state dict
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            # If there's an error, we'll try to help debug it
            logger.warning(f"Error loading state dict: {e}")
            logger.info("Checking state dict keys...")
            
            # Print some diagnostic information
            model_keys = set(model.state_dict().keys())
            state_dict_keys = set(state_dict.keys())
            
            missing_keys = model_keys - state_dict_keys
            unexpected_keys = state_dict_keys - model_keys
            
            if missing_keys:
                logger.warning(f"Missing {len(missing_keys)} keys in state dict:")
                for key in sorted(list(missing_keys))[:10]:  # Show first 10
                    logger.warning(f"  {key}")
                if len(missing_keys) > 10:
                    logger.warning(f"  ... and {len(missing_keys) - 10} more")
            
            if unexpected_keys:
                logger.warning(f"Unexpected {len(unexpected_keys)} keys in state dict:")
                for key in sorted(list(unexpected_keys))[:10]:  # Show first 10
                    logger.warning(f"  {key}")
                if len(unexpected_keys) > 10:
                    logger.warning(f"  ... and {len(unexpected_keys) - 10} more")
            
            # If we have detailed shape info
            if missing_keys:
                logger.warning("Try downloading the model again using fetch_gpt2_weights.py")
                raise RuntimeError("Failed to load model. See warnings above.")
        
        return model