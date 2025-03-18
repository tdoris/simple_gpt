from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the transformer model."""
    model_type: str = "gpt"  # Options: "transformer", "gpt"
    vocab_size: int = 50257  # GPT-2 vocabulary size
    d_model: int = 768      # Dimension of embeddings and hidden states
    num_heads: int = 12     # Number of attention heads
    num_layers: int = 12    # Number of transformer layers
    d_ff: int = 3072       # Dimension of feed-forward layer
    max_seq_length: int = 2048  # Maximum sequence length
    dropout: float = 0.1    # Dropout rate
    
    # Only used for the TransformerModel
    num_encoder_layers: Optional[int] = None
    num_decoder_layers: Optional[int] = None
    
    def __post_init__(self):
        # If not specified, set encoder and decoder layers equal to num_layers
        if self.num_encoder_layers is None:
            self.num_encoder_layers = self.num_layers
        if self.num_decoder_layers is None:
            self.num_decoder_layers = self.num_layers


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    num_train_epochs: int = 3
    max_steps: int = -1  # If > 0, overrides num_train_epochs
    warmup_ratio: float = 0.1
    fp16: bool = False  # Mixed precision training
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 2  # Limit the number of checkpoints saved
    seed: int = 42


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_length: int = 100
    min_length: int = 0
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    do_sample: bool = True
    num_return_sequences: int = 1
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    

@dataclass
class DataConfig:
    """Configuration for data processing."""
    dataset_name: str = "wikitext"
    dataset_config_name: str = "wikitext-103-raw-v1"
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    preprocessing_num_workers: int = 4
    overwrite_cache: bool = False
