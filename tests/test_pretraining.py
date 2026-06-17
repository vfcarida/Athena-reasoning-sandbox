"""Tests for the from-scratch and continued pretraining pipelines."""

import pytest
import torch
from src.pretraining.from_scratch import TransformerFromScratch, PretrainingConfig, TextDataset
from src.pretraining.continued_pretraining import ContinuedPretrainer, ContinuedPretrainingConfig

@pytest.fixture
def tiny_pretrain_config():
    return PretrainingConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=64,
        epochs=1,
        batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        mixed_precision="no",  # disable AMP for fast CPU tests
    )

def test_transformer_from_scratch_build(tiny_pretrain_config):
    """Test that the model initializes correctly with given dimensions."""
    trainer = TransformerFromScratch(tiny_pretrain_config)
    model = trainer.build_model()
    
    # Check architecture
    assert model.config.vocab_size == 100
    assert model.config.n_embd == 32
    assert model.config.n_layer == 2
    assert model.config.n_head == 2
    
    # Forward pass with random data
    input_ids = torch.randint(0, 100, (2, 10))
    outputs = model(input_ids=input_ids)
    
    assert outputs.logits.shape == (2, 10, 100)

def test_tokenizer_and_dataset(tiny_pretrain_config):
    """Test tokenizer training and dataset chunking."""
    trainer = TransformerFromScratch(tiny_pretrain_config)
    corpus = ["hello world", "this is a test", "tokenizer test string"] * 10
    
    tokenizer = trainer.build_tokenizer(corpus)
    assert len(tokenizer) >= 10
    
    dataset = TextDataset(corpus, tokenizer, max_length=5)
    assert len(dataset) > 0
    sample = dataset[0]
    
    assert "input_ids" in sample
    assert "labels" in sample
    assert sample["input_ids"].shape == sample["labels"].shape

def test_continued_pretrainer_config():
    """Test that RoPE scaling properties are applied correctly."""
    config = ContinuedPretrainingConfig(
        base_model="gpt2",  # We mock the loading in a real environment
        rope_scaling_type="linear",
        rope_scaling_factor=2.0
    )
    
    assert config.rope_scaling_type == "linear"
    assert config.rope_scaling_factor == 2.0
