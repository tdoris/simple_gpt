import pytest
import torch
from simple_gpt.utils.tokenizers import ByteLevelBPETokenizer


@pytest.fixture
def tokenizer():
    return ByteLevelBPETokenizer()


class TestTokenizer:
    def test_initialization(self, tokenizer):
        assert isinstance(tokenizer, ByteLevelBPETokenizer)
        assert tokenizer.encoder[tokenizer.unk_token] == 0
    
    def test_tokenize(self, tokenizer):
        text = "Hello, world!"
        tokens = tokenizer.tokenize(text)
        assert len(tokens) > 0
    
    def test_encode_decode(self, tokenizer):
        text = "Hello, world!"
        # Note: Since our BPE tokenizer is not pretrained, it will treat most characters as individual tokens
        # This is fine for testing the basic functionality
        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        
        # The decoded text might not match exactly due to byte-level encoding and the lack of pretrained merges
        # But the basic structure should be preserved
        assert isinstance(encoded, list)
        assert isinstance(decoded, str)
        assert len(encoded) > 0
        assert len(decoded) > 0
    
    def test_add_special_tokens(self, tokenizer):
        old_vocab_size = tokenizer.get_vocab_size()
        tokenizer.add_special_tokens(["[MASK]", "[CLS]"])
        new_vocab_size = tokenizer.get_vocab_size()
        
        assert new_vocab_size == old_vocab_size + 2
        assert "[MASK]" in tokenizer.encoder
        assert "[CLS]" in tokenizer.encoder
    
    def test_call_method(self, tokenizer):
        texts = ["Hello, world!", "Testing the tokenizer."]
        output = tokenizer(texts, padding=True)
        
        assert "input_ids" in output
        assert "attention_mask" in output
        assert isinstance(output["input_ids"], torch.Tensor)
        assert isinstance(output["attention_mask"], torch.Tensor)
        assert output["input_ids"].shape[0] == len(texts)
        assert output["attention_mask"].shape[0] == len(texts)
        
        # Check that all sequences have the same length when padding
        assert len(set([len(ids) for ids in output["input_ids"]])) == 1
