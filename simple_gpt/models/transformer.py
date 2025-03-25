import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output_layer = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = q.size(0)
        
        # Linear projections and split into heads
        q = self.query(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.key(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.value(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)  # Using a smaller value to avoid fp16 overflow
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attention_output = torch.matmul(attention_weights, v)
        
        # Concatenate heads and pass through final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.output_layer(attention_output)
        
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.gelu(self.fc1(x)))
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection and layer normalization
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        encoder_output: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None, 
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual connection and layer normalization
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention with residual connection and layer normalization
        attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerModel(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int = 512, 
        num_heads: int = 8, 
        num_encoder_layers: int = 6, 
        num_decoder_layers: int = 6, 
        d_ff: int = 2048, 
        max_seq_length: int = 2048, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_decoder_layers)
        ])
        
        self.generator = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters with Xavier/Glorot initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(
        self, 
        src: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Embed and add positional encoding
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(src)
        src = self.dropout(src)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        
        return src
    
    def decode(
        self, 
        tgt: torch.Tensor, 
        memory: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None, 
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Embed and add positional encoding
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.positional_encoding(tgt)
        tgt = self.dropout(tgt)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, src_mask, tgt_mask)
        
        return tgt
    
    def forward(
        self, 
        src: torch.Tensor, 
        tgt: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None, 
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, src_mask, tgt_mask)
        output = self.generator(output)
        return output


class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: int = 3072,
        max_seq_length: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying: Use the same weights for embedding and output layer
        self.lm_head.weight = self.token_embedding.weight
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Initialize parameters
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def get_causal_mask(self, seq_len: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return ~mask
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length = input_ids.size()
        
        if seq_length > self.max_seq_length:
            raise ValueError(f"Sequence length ({seq_length}) exceeds model's maximum length ({self.max_seq_length})")
        
        # Position indices for position embeddings
        positions = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(token_emb + pos_emb)
        
        # Create causal mask for self-attention
        causal_mask = self.get_causal_mask(seq_length).to(input_ids.device)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            causal_mask = causal_mask & attention_mask.bool().unsqueeze(1).unsqueeze(2)
        
        # Forward pass through transformer layers
        for layer in self.layers:
            x = layer(x, causal_mask)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.0,
        min_length: int = 0
    ) -> torch.Tensor:
        """
        Generate text using the model.
        
        Note: If the generation produces unexpectedly repetitive or strange tokens,
        it could mean that:
        1. The model hasn't been trained properly
        2. The vocabulary size mismatch between the tokenizer and the model
        3. Temperature is too high or too low
        4. The model output layer is misconfigured
        
        Args:
            input_ids: Input token ids
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling (higher = more random)
            top_k: Only sample from the top k most likely tokens
            top_p: Only sample from the smallest set of tokens that exceed cumulative probability p
            do_sample: Whether to sample or use greedy decoding
            eos_token_id: End of sentence token ID, to stop generation when produced
            repetition_penalty: Penalize repeated tokens (1.0 = no penalty, > 1.0 = penalty)
            min_length: Minimum length of the generated sequence
            
        Returns:
            Generated token ids
        """
        # Debug: Check vocabulary size
        vocab_size = self.lm_head.weight.size(0)
        print(f"Model vocabulary size: {vocab_size}")
        
        # Debug: Check input token range
        min_token = input_ids.min().item()
        max_token = input_ids.max().item()
        print(f"Input tokens range: {min_token} to {max_token}")
        
        if max_token >= vocab_size:
            print(f"WARNING: Input contains token IDs ({max_token}) >= vocabulary size ({vocab_size})")
        batch_size = input_ids.size(0)
        generated = input_ids.clone()
        
        # Keep track of which sequences are already finished
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        
        # Store vocabulary size for convenience
        vocab_size = self.lm_head.weight.size(0)
        
        for cur_len in range(max_new_tokens):
            if (generated.size(1) > self.max_seq_length) or (unfinished_sequences.sum() == 0):
                break
                
            # Forward pass to get logits for the next token
            with torch.no_grad():
                # Performance optimization: For long sequences, only process the most recent tokens
                # to avoid recomputing the entire sequence each time
                context_window = min(self.max_seq_length, generated.size(1))
                recent_input_ids = generated[:, -context_window:]
                
                # Create proper attention mask for the sequence - all 1s for actual tokens
                attention_mask = torch.ones(recent_input_ids.size(), device=recent_input_ids.device)
                
                # Forward pass with the attention mask
                logits = self.forward(recent_input_ids, attention_mask)
                
                # We only need the logits for the last token for next token prediction
                next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty - this helps reduce repetitive text
            if repetition_penalty != 1.0:
                # Create a tensor of zeros with same shape as the vocabulary
                penalty_tensor = torch.ones(batch_size, vocab_size, device=next_token_logits.device)
                
                # For each batch, identify which tokens have already been generated
                for i in range(batch_size):
                    for token_id in generated[i]:
                        # Apply penalty to tokens that have already been generated
                        penalty_tensor[i, token_id] = repetition_penalty
                
                # Apply penalty - divide or multiply logits based on whether they're positive or negative
                next_token_logits = torch.where(
                    next_token_logits > 0,
                    next_token_logits / penalty_tensor, 
                    next_token_logits * penalty_tensor
                )
            
            # Prevent EOS token if we haven't reached min_length yet
            if cur_len < min_length and eos_token_id is not None:
                next_token_logits[:, eos_token_id] = -float("inf")
            
            # Apply top-k filtering - improved implementation
            if top_k is not None and top_k > 0:
                # Get top-k values and indices for each batch item
                top_k_values, top_k_indices = torch.topk(
                    next_token_logits, min(top_k, next_token_logits.size(-1)), dim=-1
                )
                
                # Create new logits filled with -inf
                next_token_logits_filtered = torch.full_like(
                    next_token_logits, float("-inf")
                )
                
                # Scatter top-k values back to the full logits tensor
                next_token_logits_filtered.scatter_(
                    -1, top_k_indices, top_k_values
                )
                next_token_logits = next_token_logits_filtered
            
            # Apply top-p (nucleus) filtering - improved implementation
            if top_p is not None and top_p < 1.0:
                # Sort logits in descending order
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True, dim=-1
                )
                
                # Calculate softmax probabilities
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                
                # Calculate cumulative probabilities
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                
                # Shift the indices to the right to keep the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Create a mask for tokens to be removed
                sorted_logits[sorted_indices_to_remove] = -float("inf")
                
                # Scatter sorted logits back to original indexing
                for i in range(batch_size):
                    next_token_logits[i] = torch.index_select(
                        sorted_logits[i], dim=0, index=torch.argsort(sorted_indices[i])
                    )
            
            # Sample or take max
            if do_sample:
                # Apply softmax to convert logits to probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Make sure probabilities are valid
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    print("WARNING: Found NaN or Inf in probabilities, using greedy decoding instead")
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                else:
                    # Sample from the distribution
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Take most likely token (greedy decoding)
                next_tokens = torch.argmax(next_token_logits, dim=-1)
                
            # Debug: Print top token choices
            if next_tokens.size(0) == 1:  # Only for single sequence generation
                top_tokens = torch.topk(next_token_logits[0], min(5, next_token_logits.size(-1)))
                top_indices = top_tokens.indices.tolist()
                print(f"Top 5 token indices: {top_indices}")
            
            # Only replace tokens in unfinished sequences
            tokens_to_add = next_tokens * unfinished_sequences + (1 - unfinished_sequences) * eos_token_id \
                if eos_token_id is not None else next_tokens
            
            # Add next token to the sequences
            generated = torch.cat([generated, tokens_to_add.unsqueeze(-1)], dim=-1)
            
            # Update which sequences are finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences * (tokens_to_add != eos_token_id)
        
        return generated
