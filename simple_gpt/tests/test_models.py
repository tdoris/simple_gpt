import torch
import pytest
from simple_gpt.models import TransformerModel, GPTModel


@pytest.fixture
def transformer_model():
    return TransformerModel(
        vocab_size=1000,
        d_model=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=512,
        max_seq_length=128,
        dropout=0.1
    )


@pytest.fixture
def gpt_model():
    return GPTModel(
        vocab_size=1000,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=512,
        max_seq_length=128,
        dropout=0.1
    )


class TestTransformerModel:
    def test_initialization(self, transformer_model):
        assert isinstance(transformer_model, TransformerModel)
    
    def test_forward_pass(self, transformer_model):
        batch_size = 2
        src_seq_len = 16
        tgt_seq_len = 8
        
        src = torch.randint(0, 1000, (batch_size, src_seq_len))
        tgt = torch.randint(0, 1000, (batch_size, tgt_seq_len))
        
        src_mask = torch.ones(batch_size, src_seq_len, dtype=torch.bool)
        tgt_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len)).expand(
            batch_size, 1, tgt_seq_len, tgt_seq_len
        )
        
        output = transformer_model(src, tgt, src_mask, tgt_mask)
        
        assert output.shape == (batch_size, tgt_seq_len, 1000)
    
    def test_encoder(self, transformer_model):
        batch_size = 2
        src_seq_len = 16
        
        src = torch.randint(0, 1000, (batch_size, src_seq_len))
        src_mask = torch.ones(batch_size, src_seq_len, dtype=torch.bool)
        
        memory = transformer_model.encode(src, src_mask)
        
        assert memory.shape == (batch_size, src_seq_len, transformer_model.d_model)
    
    def test_decoder(self, transformer_model):
        batch_size = 2
        src_seq_len = 16
        tgt_seq_len = 8
        
        tgt = torch.randint(0, 1000, (batch_size, tgt_seq_len))
        memory = torch.randn(batch_size, src_seq_len, transformer_model.d_model)
        
        output = transformer_model.decode(
            tgt, memory, None, None
        )
        
        assert output.shape == (batch_size, tgt_seq_len, transformer_model.d_model)


class TestGPTModel:
    def test_initialization(self, gpt_model):
        assert isinstance(gpt_model, GPTModel)
    
    def test_forward_pass(self, gpt_model):
        batch_size = 2
        seq_len = 16
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        logits = gpt_model(input_ids, attention_mask)
        
        assert logits.shape == (batch_size, seq_len, 1000)
    
    def test_causal_mask(self, gpt_model):
        seq_len = 16
        
        mask = gpt_model.get_causal_mask(seq_len)
        
        assert mask.shape == (seq_len, seq_len)
        assert torch.all(mask.tril().bool())  # Lower triangular is all True
        assert not torch.any(mask.triu(diagonal=1).bool())  # Upper triangular (excl. diagonal) is all False
    
    def test_generate(self, gpt_model):
        batch_size = 2
        seq_len = 4
        max_new_tokens = 5
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        generated = gpt_model.generate(input_ids, max_new_tokens=max_new_tokens)
        
        assert generated.shape[0] == batch_size
        assert generated.shape[1] >= seq_len
        assert generated.shape[1] <= seq_len + max_new_tokens
        
        # Check that the original input is preserved at the beginning
        assert torch.all(generated[:, :seq_len] == input_ids)
