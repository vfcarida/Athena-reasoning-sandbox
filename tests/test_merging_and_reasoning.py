"""Tests for model merging logic and SwiReasoning engine."""

import pytest

from src.reasoning.swi_reasoning import SwiReasoningEngine, SwiReasoningSimulator
from src.utils.metrics import overthinking_index


@pytest.mark.heavy
def test_slerp():
    """Test Spherical Linear Interpolation math."""
    torch = pytest.importorskip("torch")
    from src.merging.merge_operators import TensorMergeOperators

    # Two orthogonal vectors
    v1 = torch.tensor([1.0, 0.0])
    v2 = torch.tensor([0.0, 1.0])

    # t=0.5 should give equal weights (normalized to length 1)
    # Expected: [sqrt(0.5), sqrt(0.5)] -> [0.7071, 0.7071]
    res = TensorMergeOperators.slerp(v1, v2, t=0.5)
    assert torch.allclose(res, torch.tensor([0.7071, 0.7071]), atol=1e-4)

    # t=0.0 should give v1
    res = TensorMergeOperators.slerp(v1, v2, t=0.0)
    assert torch.allclose(res, v1, atol=1e-4)


@pytest.mark.heavy
def test_dare_drop_and_rescale():
    """Test DARE probability dropping and rescaling."""
    torch = pytest.importorskip("torch")
    from src.merging.merge_operators import TensorMergeOperators

    base = torch.ones(10)
    model = torch.ones(10) * 2.0

    # drop_rate=0.5 means 50% of the delta should be set to 0.
    # The remaining 50% should be scaled by 1/(1-0.5) = 2.0
    res = TensorMergeOperators.dare_drop_and_rescale(base, model, drop_rate=0.5, seed=42)

    delta = res - base
    # Check that about 50% are zeroed
    zeros = (delta == 0.0).sum().item()
    assert 2 <= zeros <= 8  # Statistical bounds for small tensor

    # Check that non-zero deltas are scaled
    non_zeros = delta[delta != 0.0]
    # Original delta is 1.0, scaled by 2.0 -> should be 2.0
    assert torch.allclose(non_zeros, torch.tensor(2.0))


def test_swi_reasoning_metrics():
    """Test Overthinking Index calculation without PyTorch dependency."""
    res_normal = overthinking_index(thinking_tokens=10, total_tokens=100)
    assert res_normal["is_overthinking"] is False
    assert res_normal["efficiency_score"] > 0

    res_over = overthinking_index(thinking_tokens=40, total_tokens=50)
    assert res_over["is_overthinking"] is True
    assert res_over["efficiency_score"] < res_normal["efficiency_score"]


def test_swi_reasoning_simulator_sync_simulation():
    """Test synchronous SwiReasoning simulation in offline environment."""
    sim = SwiReasoningSimulator(vocab_size=100, entropy_threshold=1.5, seed=42)
    result = sim.simulate("Analyze database schema partitioning", num_steps=20)
    assert result.state.total_tokens == 20
    assert result.prompt == "Analyze database schema partitioning"
    assert len(result.state.entropy_history) == 20
    assert result.state.switch_count >= 0

    summary = result.summary()
    assert summary["total_tokens"] == 20
    assert "overthinking_ratio" in summary
    assert "entropy_avg" in summary

    plan = result.to_agent_plan()
    assert plan.task_goal == "Analyze database schema partitioning"
    assert len(plan.steps) == 1


def test_calculate_entropy_pure_python():
    """Test Shannon entropy calculation on pure-Python lists and edge cases."""
    # Uniform distribution: 2 equal outcomes -> log2(2) = 1.0 bit
    entropy_uniform = SwiReasoningEngine.calculate_entropy([2.0, 2.0])
    assert abs(entropy_uniform - 1.0) < 1e-4

    # Highly peaked distribution -> near 0 entropy
    entropy_peaked = SwiReasoningEngine.calculate_entropy([100.0, -100.0])
    assert entropy_peaked < 0.01

    # Degenerate empty input
    assert SwiReasoningEngine.calculate_entropy([]) == 0.0

