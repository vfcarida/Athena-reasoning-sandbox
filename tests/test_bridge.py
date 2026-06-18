"""Tests for the Agent-Bench Bridge."""

import pytest
import os
from src.bridge.agent_bench_bridge import BridgeConfig, AgentBenchBridge

def test_bridge_initialization():
    """Test bridge config parsing and availability logic."""
    config = BridgeConfig(
        agent_bench_path="../Agent-Bench",
        default_suite="test_suite"
    )
    bridge = AgentBenchBridge(config)
    
    assert bridge.config.default_suite == "test_suite"
    # is_available depends on environment, just ensure it doesn't crash
    assert isinstance(bridge.is_available, bool)

def test_agent_bench_metrics_list():
    """Test that all required metrics are documented."""
    bridge = AgentBenchBridge()
    metrics = bridge.get_agent_bench_metrics()
    
    assert "functional_score" in metrics
    assert "tool_correctness" in metrics
    assert "plan_quality" in metrics
    assert isinstance(metrics["functional_score"], str)

def test_export_for_evaluation_yaml(tmp_path):
    """Test that export_for_evaluation generates Agent-Bench compatible YAML."""
    import torch.nn as nn
    bridge = AgentBenchBridge()
    
    # Mock model and tokenizer
    class DummyModel(nn.Module):
        def save_pretrained(self, path):
            pass
    class DummyTokenizer:
        def save_pretrained(self, path):
            pass
            
    model = DummyModel()
    tokenizer = DummyTokenizer()
    
    out_dir = tmp_path / "export_test"
    
    bridge.export_for_evaluation(
        model=model,
        tokenizer=tokenizer,
        output_path=out_dir,
        model_name="test-model",
        provider="vllm",
        prompt_template="chatml",
        has_thinking_tokens=True,
        lineage={"parents": ["modelA"]}
    )
    
    yaml_path = out_dir / "athena_experiment.yaml"
    assert yaml_path.exists()
    
    import yaml
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
        
    assert "models" in data
    config = data["models"][0]
    assert config["model_id"] == "test-model"
    assert config["provider"] == "vllm"
    assert config["prompt_template"] == "chatml"
    assert config["tensor_parallel_size"] == 1
    assert config["has_thinking_tokens"] is True
    assert config["lineage"] == {"parents": ["modelA"]}
