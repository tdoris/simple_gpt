import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any
from datasets import load_dataset
from simple_gpt.configs import DataConfig
from transformers import PreTrainedTokenizer


class TextDataset(Dataset):
    """Dataset for loading and preprocessing text data for language model training."""
    
    def __init__(
        self, 
        texts: List[str], 
        tokenizer: PreTrainedTokenizer, 
        max_length: int = 128,  # Reduced from 1024 to make testing faster 
        stride: int = 64  # Reduced from 512 to make testing faster
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        # Tokenize all texts and create chunks
        self.examples = []
        
        # Find non-empty texts
        valid_texts = [text for text in texts if len(text.strip()) > 0]
        
        if not valid_texts:
            valid_texts = ["Hello world. This is a test sentence to ensure the dataset is not empty."]
        
        for text in valid_texts:
            tokenized = tokenizer.encode(text)
            
            # If the text is too short, just use it as is (with padding if needed)
            if len(tokenized) < max_length:
                if len(tokenized) > 1:  # Make sure there's at least 2 tokens for input/label
                    self.examples.append(tokenized)
                continue
                
            # Create chunks with specified max_length and stride
            for i in range(0, len(tokenized) - max_length + 1, stride):
                chunk = tokenized[i:i + max_length]
                if len(chunk) == max_length:
                    self.examples.append(chunk)
            
            # Include the last chunk if it's not already included and has enough tokens
            last_chunk = tokenized[-max_length:]
            if len(last_chunk) >= 2:  # Make sure there's at least 2 tokens for input/label
                self.examples.append(last_chunk)
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        input_ids = example[:-1]  # Input: all tokens except the last one
        labels = example[1:]     # Labels: all tokens except the first one
        
        attention_mask = [1] * len(input_ids)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


def prepare_dataset(config: DataConfig, tokenizer: PreTrainedTokenizer) -> Tuple[TextDataset, Optional[TextDataset]]:
    """Prepare training and validation datasets based on the configuration."""
    if config.dataset_name is not None:
        # Load dataset from HuggingFace's datasets
        raw_datasets = load_dataset(
            config.dataset_name,
            config.dataset_config_name,
            cache_dir=None
        )
        
        # Extract texts from the datasets
        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]
        
        train_texts = raw_datasets["train"][text_column_name]
        if "validation" in raw_datasets:
            eval_texts = raw_datasets["validation"][text_column_name]
        else:
            # If no validation set is available, split the train set
            train_val_split = int(0.9 * len(train_texts))
            eval_texts = train_texts[train_val_split:]
            train_texts = train_texts[:train_val_split]
    else:
        # Load from local files
        train_texts = []
        eval_texts = []
        
        if config.train_file:
            with open(config.train_file, 'r', encoding='utf-8') as f:
                train_texts = f.readlines()
        
        if config.validation_file:
            with open(config.validation_file, 'r', encoding='utf-8') as f:
                eval_texts = f.readlines()
    
    # Limit samples if requested
    if config.max_train_samples and len(train_texts) > config.max_train_samples:
        train_texts = train_texts[:config.max_train_samples]
    
    if config.max_val_samples and len(eval_texts) > config.max_val_samples:
        eval_texts = eval_texts[:config.max_val_samples]
    
    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer)
    eval_dataset = TextDataset(eval_texts, tokenizer) if eval_texts else None
    
    return train_dataset, eval_dataset


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader to handle padding."""
    # Find the maximum length in this batch
    max_input_len = max([item["input_ids"].size(0) for item in batch])
    max_label_len = max([item["labels"].size(0) for item in batch])
    
    input_ids = []
    attention_masks = []
    labels = []
    
    for item in batch:
        # Pad input_ids
        padded_input_ids = torch.nn.functional.pad(
            item["input_ids"], 
            (0, max_input_len - item["input_ids"].size(0)), 
            value=0
        )
        input_ids.append(padded_input_ids)
        
        # Pad attention_masks
        padded_attention_mask = torch.nn.functional.pad(
            item["attention_mask"], 
            (0, max_input_len - item["attention_mask"].size(0)), 
            value=0
        )
        attention_masks.append(padded_attention_mask)
        
        # Pad labels
        padded_labels = torch.nn.functional.pad(
            item["labels"], 
            (0, max_label_len - item["labels"].size(0)), 
            value=-100  # Padding tokens are ignored in loss calculation
        )
        labels.append(padded_labels)
    
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(labels)
    }


def get_dataloader(
    dataset: Dataset, 
    batch_size: int, 
    shuffle: bool = True, 
    num_workers: int = 4
) -> DataLoader:
    """Create a DataLoader for the dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )


def tokenize_function(examples, tokenizer, max_length=1024):
    """Tokenize text examples for HuggingFace datasets."""
    # Get the column name for texts
    column_names = examples.keys()
    text_column_name = "text" if "text" in column_names else list(column_names)[0]
    
    # Tokenize the texts
    tokenized = tokenizer(
        examples[text_column_name],
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Add labels for language modeling (shifted tokens)
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized


def prepare_datasets(
    raw_datasets, 
    tokenizer, 
    max_train_samples=None, 
    max_val_samples=None, 
    max_seq_length=1024
):
    """Prepare datasets from HuggingFace raw datasets."""
    # Extract texts from the datasets
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    
    # Limit samples if requested
    if max_train_samples is not None:
        train_dataset = raw_datasets["train"].select(range(min(max_train_samples, len(raw_datasets["train"]))))
    else:
        train_dataset = raw_datasets["train"]
    
    if "validation" in raw_datasets:
        if max_val_samples is not None:
            eval_dataset = raw_datasets["validation"].select(range(min(max_val_samples, len(raw_datasets["validation"]))))
        else:
            eval_dataset = raw_datasets["validation"]
    else:
        # If no validation set is available, split the train set
        train_val_split = int(0.9 * len(train_dataset))
        eval_dataset = train_dataset.select(range(train_val_split, len(train_dataset)))
        train_dataset = train_dataset.select(range(train_val_split))
    
    # Create the TextDataset objects
    train_texts = train_dataset[text_column_name]
    eval_texts = eval_dataset[text_column_name]
    
    train_dataset = TextDataset(train_texts, tokenizer, max_length=max_seq_length)
    eval_dataset = TextDataset(eval_texts, tokenizer, max_length=max_seq_length)
    
    return train_dataset, eval_dataset