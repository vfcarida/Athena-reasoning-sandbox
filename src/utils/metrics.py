"""Shared statistical metrics for NeuroFusionBench.

This module provides reusable mathematical functions used across the framework:
- Shannon entropy for information-theoretic analysis
- Elo rating for comparative model evaluation
- Overthinking index for inference efficiency measurement
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
import torch


def shannon_entropy(
    probabilities: Union[torch.Tensor, np.ndarray, list[float]],
    base: float = 2.0,
    eps: float = 1e-12,
) -> float:
    """Compute the Shannon entropy of a discrete probability distribution.

    The Shannon entropy quantifies the average information content (uncertainty)
    of a random variable:

        H(X) = −Σᵢ P(xᵢ) · log_b(P(xᵢ))

    Higher entropy indicates greater uncertainty in the distribution; lower
    entropy indicates higher confidence.

    Args:
        probabilities: A 1-D array or tensor of probabilities. Does not need
            to be pre-normalized — the function will normalize to sum=1 if needed.
        base: Logarithmic base. Default: 2.0 (entropy measured in bits).
            Use ``math.e`` for nats or 10.0 for hartleys.
        eps: Small epsilon to clamp near-zero probabilities and avoid log(0).

    Returns:
        Scalar entropy value (float). Returns 0.0 for degenerate distributions.

    Raises:
        ValueError: If ``probabilities`` contains negative values.

    Example:
        >>> shannon_entropy([0.25, 0.25, 0.25, 0.25])  # Maximum entropy for 4 classes
        2.0
        >>> shannon_entropy([1.0, 0.0, 0.0, 0.0])       # Minimum entropy (certain)
        0.0
    """
    # Convert to numpy for unified processing
    if isinstance(probabilities, torch.Tensor):
        probs = probabilities.detach().cpu().to(torch.float64).numpy()
    elif isinstance(probabilities, list):
        probs = np.array(probabilities, dtype=np.float64)
    else:
        probs = np.asarray(probabilities, dtype=np.float64)

    # Flatten to 1-D
    probs = probs.flatten()

    if np.any(probs < 0):
        raise ValueError("Probabilities must be non-negative.")

    # Handle degenerate case: all zeros
    total = probs.sum()
    if total < eps:
        return 0.0

    # Normalize to a valid probability distribution
    probs = probs / total

    # Clamp and compute entropy
    probs = np.clip(probs, eps, 1.0)
    log_probs = np.log(probs) / np.log(base)
    entropy = -np.sum(probs * log_probs)

    return float(max(entropy, 0.0))


def elo_rating(
    wins: int,
    losses: int,
    draws: int = 0,
    k_factor: float = 32.0,
    initial_rating: float = 1500.0,
    opponent_rating: float = 1500.0,
) -> dict[str, float]:
    """Compute Elo rating update after a series of matches.

    The Elo system provides a relative performance measure where the expected
    score is:

        E = 1 / (1 + 10^((R_opponent − R_self) / 400))

    And the rating update is:

        R' = R + K · (S − E)

    where S is the actual score (1 for win, 0.5 for draw, 0 for loss).

    Args:
        wins: Number of wins.
        losses: Number of losses.
        draws: Number of draws. Default: 0.
        k_factor: Sensitivity factor for rating updates. Higher values cause
            more volatile ratings. Default: 32.0.
        initial_rating: Starting Elo rating. Default: 1500.0.
        opponent_rating: Average opponent Elo rating. Default: 1500.0.

    Returns:
        Dictionary containing:
            - ``final_rating``: Updated Elo rating after all matches.
            - ``expected_score``: Expected win probability per match.
            - ``actual_score``: Observed win rate across all matches.
            - ``total_matches``: Total number of matches played.

    Raises:
        ValueError: If any count is negative.
    """
    if wins < 0 or losses < 0 or draws < 0:
        raise ValueError("Match counts must be non-negative.")

    total_matches = wins + losses + draws
    if total_matches == 0:
        return {
            "final_rating": initial_rating,
            "expected_score": 0.5,
            "actual_score": 0.0,
            "total_matches": 0,
        }

    # Expected score per match (probability of winning against opponent)
    exponent = (opponent_rating - initial_rating) / 400.0
    expected_score = 1.0 / (1.0 + math.pow(10.0, exponent))

    # Actual score: 1 per win, 0.5 per draw, 0 per loss
    actual_total = wins * 1.0 + draws * 0.5 + losses * 0.0
    actual_score = actual_total / total_matches

    # Update rating iteratively for each match
    rating = initial_rating
    for _ in range(wins):
        e = 1.0 / (1.0 + math.pow(10.0, (opponent_rating - rating) / 400.0))
        rating += k_factor * (1.0 - e)

    for _ in range(draws):
        e = 1.0 / (1.0 + math.pow(10.0, (opponent_rating - rating) / 400.0))
        rating += k_factor * (0.5 - e)

    for _ in range(losses):
        e = 1.0 / (1.0 + math.pow(10.0, (opponent_rating - rating) / 400.0))
        rating += k_factor * (0.0 - e)

    return {
        "final_rating": round(rating, 2),
        "expected_score": round(expected_score, 4),
        "actual_score": round(actual_score, 4),
        "total_matches": total_matches,
    }


def overthinking_index(
    thinking_tokens: int,
    total_tokens: int,
    max_acceptable_ratio: float = 0.5,
) -> dict[str, float]:
    """Compute the Overthinking Index for inference efficiency analysis.

    Measures the proportion of generated tokens spent in latent "thinking" mode
    relative to total output. A high index suggests the model is allocating
    excessive compute to internal reasoning without proportional output quality
    gains.

    Formulation:
        OI = thinking_tokens / total_tokens
        Efficiency = 1.0 − min(OI / max_acceptable_ratio, 1.0)

    Args:
        thinking_tokens: Number of tokens generated in latent thinking mode.
        total_tokens: Total number of tokens generated (thinking + explicit).
        max_acceptable_ratio: The ratio above which overthinking is considered
            critical. Used to compute the efficiency score. Default: 0.5.

    Returns:
        Dictionary containing:
            - ``overthinking_ratio``: Raw ratio of thinking to total tokens.
            - ``efficiency_score``: Normalized efficiency in [0, 1]. 1.0 means
              no overthinking, 0.0 means critical overthinking.
            - ``is_overthinking``: Boolean flag if ratio exceeds max_acceptable_ratio.

    Raises:
        ValueError: If total_tokens is zero or negative.
        ValueError: If thinking_tokens exceeds total_tokens.
    """
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive. Got {total_tokens}.")
    if thinking_tokens < 0:
        raise ValueError(f"thinking_tokens must be non-negative. Got {thinking_tokens}.")
    if thinking_tokens > total_tokens:
        raise ValueError(
            f"thinking_tokens ({thinking_tokens}) cannot exceed "
            f"total_tokens ({total_tokens})."
        )

    ratio = thinking_tokens / total_tokens
    efficiency = 1.0 - min(ratio / max_acceptable_ratio, 1.0)

    return {
        "overthinking_ratio": round(ratio, 4),
        "efficiency_score": round(efficiency, 4),
        "is_overthinking": ratio > max_acceptable_ratio,
    }
