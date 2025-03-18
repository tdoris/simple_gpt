import os
import json
import torch
import regex as re
from typing import Dict, List, Optional, Tuple, Union
from transformers import PreTrainedTokenizer


class ByteLevelBPETokenizer:
    """A simplified version of byte-level BPE tokenizer similar to GPT-2's tokenizer."""
    
    def __init__(
        self, 
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        unk_token: str = "<|endoftext|>",
        bos_token: str = "<|endoftext|>",
        eos_token: str = "<|endoftext|>",
        pad_token: Optional[str] = None
    ):
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        
        # Initialize with empty vocabulary and merges
        self.encoder = {}
        self.decoder = {}
        self.bpe_ranks = {}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # Load pretrained tokenizer if files are provided
        if vocab_file is not None and merges_file is not None:
            self.load_tokenizer(vocab_file, merges_file)
        
        # Add special tokens to the vocabulary
        if self.unk_token not in self.encoder:
            self.add_special_tokens([self.unk_token])
        if self.bos_token != self.unk_token and self.bos_token not in self.encoder:
            self.add_special_tokens([self.bos_token])
        if self.eos_token != self.unk_token and self.eos_token not in self.encoder:
            self.add_special_tokens([self.eos_token])
        if self.pad_token is not None and self.pad_token not in self.encoder:
            self.add_special_tokens([self.pad_token])
    
    def load_tokenizer(self, vocab_file: str, merges_file: str):
        """Load tokenizer vocabulary and merges from files."""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        with open(merges_file, 'r', encoding='utf-8') as f:
            bpe_merges = f.read().split('\n')
            bpe_merges = [tuple(merge.split()) for merge in bpe_merges if merge != '']
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
    
    def save_tokenizer(self, save_directory: str):
        """Save tokenizer vocabulary and merges to files."""
        os.makedirs(save_directory, exist_ok=True)
        
        vocab_file = os.path.join(save_directory, "vocab.json")
        merges_file = os.path.join(save_directory, "merges.txt")
        
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.encoder, f, ensure_ascii=False)
        
        merges = list(self.bpe_ranks.keys())
        merges = [' '.join(merge) for merge in merges]
        with open(merges_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(merges))
    
    def add_special_tokens(self, special_tokens: List[str]):
        """Add special tokens to the vocabulary."""
        for token in special_tokens:
            if token not in self.encoder:
                self.encoder[token] = len(self.encoder)
                self.decoder[len(self.decoder)] = token
    
    def get_vocab_size(self) -> int:
        """Get the size of the vocabulary."""
        return len(self.encoder)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize a text into BPE tokens."""
        bpe_tokens = []
        for token in re.findall(r'\S+|\n+| +', text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_result = self._bpe(token)
            bpe_tokens.extend(bpe_result.split(' '))
        return bpe_tokens
    
    def _bpe(self, token: str) -> str:
        """Apply Byte-Pair Encoding to a token."""
        if not self.bpe_ranks:
            return token
        
        word = list(token)
        pairs = get_pairs(word)
        
        if not pairs:
            return token
        
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        
        return ' '.join(word)
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs."""
        ids = []
        for token in tokens:
            if token in self.encoder:
                ids.append(self.encoder[token])
            else:
                ids.append(self.encoder[self.unk_token])
        return ids
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert IDs to tokens."""
        tokens = []
        for idx in ids:
            if idx in self.decoder:
                tokens.append(self.decoder[idx])
            else:
                tokens.append(self.unk_token)
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode a text into token IDs."""
        tokens = self.tokenize(text)
        if add_special_tokens and self.bos_token:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        return self.convert_tokens_to_ids(tokens)
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs back to text."""
        tokens = self.convert_ids_to_tokens(token_ids)
        if skip_special_tokens:
            tokens = [token for token in tokens 
                     if token not in [self.unk_token, self.bos_token, self.eos_token] 
                     and (self.pad_token is None or token != self.pad_token)]
        
        text = ''.join(tokens)
        
        # Replace byte-level encoded characters with their UTF-8 representation
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors='replace')
        
        return text
    
    def __call__(self, texts: Union[str, List[str]], padding: bool = False, max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Process text(s) and return a batch of tensors with input_ids, attention_mask, etc."""
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize all texts
        all_input_ids = [self.encode(text) for text in texts]
        
        # Handle padding if requested
        if padding:
            max_len = max([len(ids) for ids in all_input_ids])
            if max_length is not None:
                max_len = min(max_len, max_length)
            
            pad_id = self.encoder[self.pad_token] if self.pad_token else 0
            
            for i, ids in enumerate(all_input_ids):
                if len(ids) < max_len:
                    all_input_ids[i] = ids + [pad_id] * (max_len - len(ids))
                elif len(ids) > max_len:
                    all_input_ids[i] = ids[:max_len]
            
            attention_mask = [[1] * len(ids[:max_len]) + [0] * (max_len - len(ids[:max_len])) for ids in all_input_ids]
        else:
            attention_mask = [[1] * len(ids) for ids in all_input_ids]
        
        return {
            "input_ids": torch.tensor(all_input_ids),
            "attention_mask": torch.tensor(attention_mask)
        }


def bytes_to_unicode():
    """Returns a mapping from bytes to unicode strings."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word."""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def load_tokenizer(tokenizer_path: Optional[str] = None) -> Union[ByteLevelBPETokenizer, PreTrainedTokenizer]:
    """Load a tokenizer either from a local path or use a pretrained one."""
    if tokenizer_path is None:
        # Use HuggingFace's tokenizer
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    else:
        # Load local tokenizer
        vocab_file = os.path.join(tokenizer_path, "vocab.json")
        merges_file = os.path.join(tokenizer_path, "merges.txt")
        return ByteLevelBPETokenizer(vocab_file=vocab_file, merges_file=merges_file)
