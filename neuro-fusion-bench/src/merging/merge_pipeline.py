"""Merge Pipeline — YAML-driven orchestrator for model weight fusion.

This module provides a high-level interface that reads Mergekit-compatible YAML
configurations and dispatches the appropriate tensor fusion algorithm from
:mod:`merge_operators`.

It abstracts the low-level tensor operations behind a configuration-driven
workflow suitable for automated experimentation pipelines.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

from .merge_operators import TensorMergeOperators

logger = logging.getLogger(__name__)


class MergePipeline:
    """Configuration-driven pipeline for executing model weight merges.

    Reads a YAML configuration file specifying the merge method, source model
    parameters, and algorithm-specific hyperparameters, then dispatches the
    appropriate :class:`TensorMergeOperators` method.

    Example:
        >>> pipeline = MergePipeline()
        >>> config = pipeline.load_config("configs/merge_config.yaml")
        >>> merged_state = pipeline.execute(config, model_states)
        >>> pipeline.save_merged(merged_state, "merged_model/pytorch_model.bin")
    """

    # Supported merge methods mapped to their handler methods
    SUPPORTED_METHODS = {"slerp", "ties", "dare"}

    def __init__(self, device: Optional[str] = None) -> None:
        """Initialize the merge pipeline.

        Args:
            device: Target device for tensor operations ("cpu", "cuda", "cuda:0").
                If None, auto-detects CUDA availability.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info("MergePipeline initialized on device: %s", self.device)

    def load_config(self, config_path: str | Path) -> dict[str, Any]:
        """Load and validate a merge configuration from a YAML file.

        The YAML file should follow the Mergekit-compatible schema with at least
        a ``merge_method`` key and the appropriate algorithm parameters.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            Parsed configuration dictionary.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the merge method is unsupported or required keys are missing.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Validate required keys
        merge_method = config.get("merge_method")
        if merge_method is None:
            raise ValueError("Configuration must specify 'merge_method'.")

        if merge_method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported merge method '{merge_method}'. "
                f"Supported: {self.SUPPORTED_METHODS}"
            )

        logger.info("Loaded configuration: method=%s from %s", merge_method, path)
        return config

    def execute(
        self,
        config: dict[str, Any],
        model_states: dict[str, list[torch.Tensor] | torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Execute the merge operation according to the loaded configuration.

        Dispatches to the appropriate fusion algorithm based on the ``merge_method``
        key in the configuration. Processes each parameter key in the model state
        dictionaries independently.

        Args:
            config: Parsed YAML configuration from :meth:`load_config`.
            model_states: Dictionary with the following structure:
                - ``"base"``: Base model state dict (``dict[str, torch.Tensor]``)
                  or a single tensor for simple merges.
                - ``"models"``: List of fine-tuned model state dicts or tensors.
                  For SLERP, only one additional model is needed.
                  For TIES, multiple models can be provided.
                  For DARE, one fine-tuned model is expected.

        Returns:
            Merged state dictionary mapping parameter names to merged tensors.

        Raises:
            ValueError: If model_states structure is incompatible with the method.
        """
        method = config["merge_method"]
        logger.info("Executing merge: method=%s", method)

        base = model_states.get("base")
        models = model_states.get("models", [])

        if base is None:
            raise ValueError("model_states must contain a 'base' key.")

        # Handle single-tensor inputs (for simple demonstrations)
        if isinstance(base, torch.Tensor):
            return self._execute_single_tensor(method, config, base, models)

        # Handle state dict inputs (full model merging)
        return self._execute_state_dict(method, config, base, models)

    def _execute_single_tensor(
        self,
        method: str,
        config: dict[str, Any],
        base: torch.Tensor,
        models: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Execute merge on single tensors (simplified API for demos)."""
        base = base.to(self.device)
        result: torch.Tensor

        if method == "slerp":
            if not models:
                raise ValueError("SLERP requires at least one model tensor to merge.")
            params = config.get("slerp", {})
            t = params.get("interpolation_factor", 0.5)
            dot_threshold = params.get("dot_threshold", 0.9995)
            model_tensor = models[0].to(self.device)
            result = TensorMergeOperators.slerp(base, model_tensor, t, dot_threshold)

        elif method == "ties":
            if not models:
                raise ValueError("TIES requires at least one task model tensor.")
            params = config.get("ties", {})
            threshold = params.get("density_threshold", 0.2)
            task_tensors = [m.to(self.device) for m in models]
            result = TensorMergeOperators.ties_merge(base, task_tensors, threshold)

        elif method == "dare":
            if not models:
                raise ValueError("DARE requires one fine-tuned model tensor.")
            params = config.get("dare", {})
            drop_rate = params.get("drop_rate", 0.3)
            seed = params.get("seed", None)
            model_tensor = models[0].to(self.device)
            result = TensorMergeOperators.dare_drop_and_rescale(
                base, model_tensor, drop_rate, seed
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        return {"merged_tensor": result}

    def _execute_state_dict(
        self,
        method: str,
        config: dict[str, Any],
        base: dict[str, torch.Tensor],
        models: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Execute merge across all parameter keys in model state dicts."""
        merged_state: dict[str, torch.Tensor] = {}
        param_keys = list(base.keys())

        logger.info("Merging %d parameter keys...", len(param_keys))

        for key in param_keys:
            base_param = base[key].to(self.device)

            if method == "slerp":
                params = config.get("slerp", {})
                t = params.get("interpolation_factor", 0.5)
                dot_threshold = params.get("dot_threshold", 0.9995)
                if models and key in models[0]:
                    other_param = models[0][key].to(self.device)
                    merged_state[key] = TensorMergeOperators.slerp(
                        base_param, other_param, t, dot_threshold
                    )
                else:
                    merged_state[key] = base_param

            elif method == "ties":
                params = config.get("ties", {})
                threshold = params.get("density_threshold", 0.2)
                task_params = [
                    m[key].to(self.device) for m in models if key in m
                ]
                if task_params:
                    merged_state[key] = TensorMergeOperators.ties_merge(
                        base_param, task_params, threshold
                    )
                else:
                    merged_state[key] = base_param

            elif method == "dare":
                params = config.get("dare", {})
                drop_rate = params.get("drop_rate", 0.3)
                seed = params.get("seed", None)
                if models and key in models[0]:
                    finetuned_param = models[0][key].to(self.device)
                    merged_state[key] = TensorMergeOperators.dare_drop_and_rescale(
                        base_param, finetuned_param, drop_rate, seed
                    )
                else:
                    merged_state[key] = base_param

        logger.info("Merge complete: %d parameters merged.", len(merged_state))
        return merged_state

    @staticmethod
    def save_merged(
        state_dict: dict[str, torch.Tensor],
        output_path: str | Path,
    ) -> Path:
        """Serialize merged model parameters to disk.

        Args:
            state_dict: Dictionary of merged parameter tensors.
            output_path: Destination file path (typically .bin or .pt extension).

        Returns:
            Resolved path to the saved file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, path)
        logger.info("Saved merged parameters to: %s", path)
        return path.resolve()
