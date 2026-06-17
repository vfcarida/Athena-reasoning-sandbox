"""Tests for model merging logic and SwiReasoning engine."""

import pytest
import torch
from src.merging.merge_operators import TensorMergeOperators
from src.reasoning.swi_reasoning import SwiReasoningSimulator
from src.utils.metrics import overthinking_index

def test_slerp():
    """Test Spherical Linear Interpolation math."""
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

def test_dare_drop_and_rescale():
    """Test DARE probability dropping and rescaling."""
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
    """Test Overthinking Index calculation."""
    res_normal = overthinking_index(thinking_tokens=10, total_tokens=100)
    assert res_normal["is_overthinking"] is False
    assert res_normal["efficiency_score"] > 0
    
    res_over = overthinking_index(thinking_tokens=40, total_tokens=50)
    assert res_over["is_overthinking"] is True
    assert res_over["efficiency_score"] < res_normal["efficiency_score"]
