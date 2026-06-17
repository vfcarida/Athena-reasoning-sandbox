"""Tensor Merge Operators — Mathematical fusion algorithms for neural weight tensors.

This module implements three state-of-the-art model merging techniques as pure
PyTorch tensor operations:

- **SLERP** (Spherical Linear Interpolation): Geometrically correct interpolation
  on the hypersphere of normalized weight vectors.
- **TIES-Merging** (Trim, Elect Sign & Merge): Redundancy-aware merging that
  resolves sign conflicts via majority consensus.
- **DARE** (Drop And REscale): Stochastic delta pruning with statistical rescaling
  to maintain the base model's activation scale.

All operations are device-agnostic and automatically handle CPU/CUDA placement.

References:
    - Shoemake, K. (1985). Animating rotation with quaternion curves. SIGGRAPH.
    - Yadav et al. (2023). TIES-Merging: Resolving Interference When Merging Models.
    - Yu et al. (2023). Language Model Merging by Drop and Rescale.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


class TensorMergeOperators:
    """Collection of static methods for neural weight tensor fusion.

    Each method operates directly on PyTorch tensors and returns a new tensor
    containing the merged parameters. All methods preserve the original tensor
    shapes and ensure device consistency.

    Example:
        >>> p0 = torch.randn(768, 768)
        >>> p1 = torch.randn(768, 768)
        >>> merged = TensorMergeOperators.slerp(p0, p1, t=0.5)
        >>> assert merged.shape == p0.shape
    """

    @staticmethod
    def slerp(
        p0: torch.Tensor,
        p1: torch.Tensor,
        t: float,
        dot_threshold: float = 0.9995,
    ) -> torch.Tensor:
        """Spherical Linear Interpolation (SLERP) between two weight tensors.

        Treats flattened weight tensors as high-dimensional vectors on the unit
        hypersphere and interpolates along the great-circle arc between them.

        Mathematical formulation:
            Slerp(p₀, p₁; t) = [sin((1-t)θ) / sin(θ)] · p₀ + [sin(tθ) / sin(θ)] · p₁

        where θ = arccos(p̂₀ · p̂₁) is the angle between the L2-normalized vectors.

        When the vectors are nearly colinear (sin(θ) ≈ 0), the method gracefully
        falls back to standard linear interpolation to avoid numerical instability.

        Args:
            p0: First weight tensor (any shape). Will be flattened internally.
            p1: Second weight tensor. Must have the same shape as ``p0``.
            t: Interpolation factor in [0, 1]. t=0 returns p0, t=1 returns p1.
            dot_threshold: Cosine similarity threshold above which the method
                falls back to linear interpolation. Default: 0.9995.

        Returns:
            Merged tensor with the same shape and device as the inputs.

        Raises:
            ValueError: If ``p0`` and ``p1`` have different shapes.
            ValueError: If ``t`` is outside [0, 1].
        """
        if p0.shape != p1.shape:
            raise ValueError(
                f"Tensor shapes must match for SLERP. "
                f"Got p0={p0.shape}, p1={p1.shape}."
            )
        if not 0.0 <= t <= 1.0:
            raise ValueError(f"Interpolation factor t must be in [0, 1]. Got t={t}.")

        original_shape = p0.shape
        device = p0.device

        # Ensure both tensors reside on the same device
        p1 = p1.to(device)

        # Flatten to 1-D vectors for geometric operations
        v0 = p0.flatten().to(torch.float32)
        v1 = p1.flatten().to(torch.float32)

        # L2-normalize to place vectors on the unit hypersphere
        norm_v0 = torch.linalg.norm(v0)
        norm_v1 = torch.linalg.norm(v1)

        if norm_v0 < 1e-12 or norm_v1 < 1e-12:
            # Degenerate case: one or both tensors are near-zero
            return ((1.0 - t) * p0 + t * p1).to(p0.dtype)

        v0_unit = v0 / norm_v0
        v1_unit = v1 / norm_v1

        # Compute cosine of the angle between normalized vectors
        dot = torch.clamp(torch.dot(v0_unit, v1_unit), -1.0, 1.0)

        # Fallback: if vectors are nearly colinear, use linear interpolation
        if torch.abs(dot) > dot_threshold:
            result = (1.0 - t) * v0 + t * v1
            return result.reshape(original_shape).to(p0.dtype)

        # Compute the angle θ and the SLERP coefficients
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)

        coeff_0 = torch.sin((1.0 - t) * theta) / sin_theta
        coeff_1 = torch.sin(t * theta) / sin_theta

        # Interpolate on the hypersphere, then scale by interpolated norms
        interpolated_norm = (1.0 - t) * norm_v0 + t * norm_v1
        result = (coeff_0 * v0_unit + coeff_1 * v1_unit) * interpolated_norm

        return result.reshape(original_shape).to(p0.dtype)

    @staticmethod
    def ties_merge(
        base_params: torch.Tensor,
        task_params_list: list[torch.Tensor],
        threshold: float = 0.2,
    ) -> torch.Tensor:
        """TIES-Merging: Trim, Elect Sign & Merge for multi-task weight fusion.

        This algorithm addresses parameter interference in model merging through
        a three-step procedure:

        1. **Trim**: Compute deltas (task_params − base_params) and zero out the
           bottom ``threshold`` fraction by magnitude, removing low-signal noise.
        2. **Elect Sign**: For each parameter position, resolve sign conflicts
           across tasks via majority vote (the sign with the greatest aggregate
           magnitude wins).
        3. **Merge**: Average the surviving deltas (those aligned with the elected
           sign) and add them back to the base parameters.

        Args:
            base_params: The base (pretrained) model's weight tensor.
            task_params_list: List of fine-tuned weight tensors from different tasks.
                Each must have the same shape as ``base_params``.
            threshold: Fraction of lowest-magnitude deltas to prune per task
                tensor. Must be in [0, 1). Default: 0.2 (prune bottom 20%).

        Returns:
            Merged tensor with the same shape and device as ``base_params``.

        Raises:
            ValueError: If any task tensor shape doesn't match ``base_params``.
            ValueError: If ``task_params_list`` is empty.
        """
        if not task_params_list:
            raise ValueError("task_params_list must contain at least one tensor.")

        for i, tp in enumerate(task_params_list):
            if tp.shape != base_params.shape:
                raise ValueError(
                    f"Task tensor {i} shape {tp.shape} does not match "
                    f"base shape {base_params.shape}."
                )

        device = base_params.device
        dtype = base_params.dtype
        base = base_params.flatten().to(torch.float32)

        # -----------------------------------------------------------------
        # Step 1: TRIM — Compute deltas and prune low-magnitude entries
        # -----------------------------------------------------------------
        trimmed_deltas: list[torch.Tensor] = []
        for tp in task_params_list:
            delta = tp.flatten().to(torch.float32).to(device) - base
            abs_delta = torch.abs(delta)

            if threshold > 0.0:
                # Compute the magnitude threshold via quantile
                quantile_val = torch.quantile(abs_delta, threshold)
                # Zero out entries below the threshold
                mask = abs_delta >= quantile_val
                delta = delta * mask.to(delta.dtype)

            trimmed_deltas.append(delta)

        # Stack deltas: shape (num_tasks, num_params)
        delta_stack = torch.stack(trimmed_deltas, dim=0)

        # -----------------------------------------------------------------
        # Step 2: ELECT SIGN — Majority vote on the sign of each parameter
        # -----------------------------------------------------------------
        # Aggregate signed magnitudes across tasks
        positive_mass = torch.sum(
            torch.where(delta_stack > 0, delta_stack, torch.zeros_like(delta_stack)),
            dim=0,
        )
        negative_mass = torch.sum(
            torch.where(delta_stack < 0, torch.abs(delta_stack), torch.zeros_like(delta_stack)),
            dim=0,
        )

        # The elected sign is the one with greater aggregate magnitude
        # +1 if positive mass dominates, -1 if negative mass dominates
        elected_sign = torch.where(
            positive_mass >= negative_mass,
            torch.ones_like(positive_mass),
            -torch.ones_like(positive_mass),
        )

        # -----------------------------------------------------------------
        # Step 3: MERGE — Average deltas that agree with the elected sign
        # -----------------------------------------------------------------
        # Mask out deltas that disagree with the elected sign
        sign_match = (torch.sign(delta_stack) == elected_sign.unsqueeze(0))
        # Also include zero deltas (they were pruned and shouldn't contribute)
        nonzero_mask = delta_stack != 0.0
        valid_mask = sign_match & nonzero_mask

        # Sum of matching deltas and count of contributors
        masked_deltas = delta_stack * valid_mask.to(delta_stack.dtype)
        sum_deltas = torch.sum(masked_deltas, dim=0)
        count = torch.sum(valid_mask.to(torch.float32), dim=0)

        # Avoid division by zero: where no task contributes, delta stays zero
        avg_deltas = torch.where(
            count > 0,
            sum_deltas / count,
            torch.zeros_like(sum_deltas),
        )

        # Apply merged deltas to the base parameters
        merged = base + avg_deltas
        return merged.reshape(base_params.shape).to(dtype)

    @staticmethod
    def dare_drop_and_rescale(
        base_params: torch.Tensor,
        finetuned_params: torch.Tensor,
        drop_rate: float = 0.3,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """DARE: Drop And REscale delta merging.

        Applies stochastic pruning to the fine-tuning deltas and rescales the
        surviving parameters to preserve the statistical scale of the base model.

        Procedure:
            1. Compute delta: Δ = finetuned − base
            2. Generate a Bernoulli mask M with P(Mᵢ = 0) = drop_rate
            3. Apply masked delta: Δ' = Δ ⊙ M
            4. Rescale survivors: Δ'' = Δ' / (1 − drop_rate)
            5. Return: base + Δ''

        The rescaling in step 4 ensures that the expected sum of deltas is
        preserved despite the random pruning, maintaining activation magnitudes.

        Args:
            base_params: The base (pretrained) model's weight tensor.
            finetuned_params: The fine-tuned model's weight tensor. Must have
                the same shape as ``base_params``.
            drop_rate: Probability of dropping each delta parameter. Must be
                in [0, 1). Default: 0.3.
            seed: Optional random seed for reproducibility. If None, the mask
                is generated non-deterministically.

        Returns:
            Merged tensor with the same shape and device as ``base_params``.

        Raises:
            ValueError: If tensor shapes don't match.
            ValueError: If ``drop_rate`` is outside [0, 1).
        """
        if base_params.shape != finetuned_params.shape:
            raise ValueError(
                f"Tensor shapes must match for DARE. "
                f"Got base={base_params.shape}, finetuned={finetuned_params.shape}."
            )
        if not 0.0 <= drop_rate < 1.0:
            raise ValueError(
                f"drop_rate must be in [0, 1). Got drop_rate={drop_rate}."
            )

        device = base_params.device
        dtype = base_params.dtype

        # Ensure device consistency
        finetuned_params = finetuned_params.to(device)

        # Compute the fine-tuning delta in float32 for numerical stability
        base_f32 = base_params.to(torch.float32)
        finetuned_f32 = finetuned_params.to(torch.float32)
        delta = finetuned_f32 - base_f32

        # Generate Bernoulli mask (1 = keep, 0 = drop)
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
        else:
            generator = None

        # Create mask on the correct device
        keep_mask = torch.bernoulli(
            torch.full(delta.shape, 1.0 - drop_rate, device=device),
            generator=generator,
        )

        # Apply mask and rescale survivors
        masked_delta = delta * keep_mask

        if drop_rate > 0.0:
            rescale_factor = 1.0 / (1.0 - drop_rate)
            masked_delta = masked_delta * rescale_factor

        # Add the rescaled delta back to the base
        merged = base_f32 + masked_delta
        return merged.to(dtype)
