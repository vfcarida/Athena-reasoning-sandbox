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
