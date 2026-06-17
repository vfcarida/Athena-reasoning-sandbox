"""Tests for SFT and LoRA fine-tuning pipelines."""

import pytest
from src.finetuning.sft_trainer import SFTOrchestrator, SFTConfig
from src.finetuning.lora_trainer import LoRAFineTuner, LoRAConfig

def test_sft_format_alpaca():
    """Test formatting samples into Alpaca template."""
    config = SFTConfig(dataset_format="alpaca")
    orchestrator = SFTOrchestrator(config)
    
    sample = {
        "instruction": "What is 2+2?",
        "input": "",
        "output": "4"
    }
    
    formatted = orchestrator.format_sample(sample)
    assert "### Instruction:\nWhat is 2+2?" in formatted
    assert "### Response:\n4" in formatted
    assert "### Input:" not in formatted # Should use NO_INPUT template

def test_sft_format_chatml():
    """Test formatting samples into ChatML template."""
    config = SFTConfig(dataset_format="chatml")
    orchestrator = SFTOrchestrator(config)
    
    sample = {
        "system": "You are a math bot.",
        "instruction": "What is 2+2?",
        "output": "4"
    }
    
    formatted = orchestrator.format_sample(sample)
    assert "<|im_start|>system\nYou are a math bot." in formatted
    assert "<|im_start|>user\nWhat is 2+2?" in formatted
    assert "<|im_start|>assistant\n4" in formatted

def test_lora_config():
    """Test LoRA configuration properties."""
    config = LoRAConfig(rank=8, alpha=16, quantize_4bit=True)
    
    assert config.rank == 8
    assert config.alpha == 16
    # Effective scaling should be alpha / rank = 2.0
    assert config.quantize_4bit is True
    assert "q_proj" in config.target_modules
