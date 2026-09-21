"""Pytest harness configuration for Athena Reasoning Sandbox.

Configures test lanes:
- Lane 1: Lightweight offline test suite (no torch/AWS/paid APIs).
- Lane 2: Heavy test suite marked with @pytest.mark.heavy.
"""

from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from unittest.mock import MagicMock

import pytest

try:
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

    class HeavyFinder:
        """Meta-path finder to stub heavy ML/GPU libraries if absent during collection."""

        HEAVY_PACKAGES = (
            "torch",
            "transformers",
            "peft",
            "trl",
            "datasets",
            "tokenizers",
            "accelerate",
        )

        def find_spec(
            self,
            fullname: str,
            path: list[str] | None,
            target: object = None,
        ) -> ModuleSpec | None:
            for pkg in self.HEAVY_PACKAGES:
                if fullname == pkg or fullname.startswith(pkg + "."):
                    return ModuleSpec(fullname, self, is_package=True)
            return None

        def create_module(self, spec: ModuleSpec) -> object:
            mod = MagicMock()
            mod.__name__ = spec.name
            mod.__path__ = []
            if spec.name == "torch":
                mod.Tensor = MagicMock
            elif spec.name == "torch.nn":
                mod.Module = object
            elif spec.name == "torch.utils.data":
                mod.Dataset = object
                mod.DataLoader = MagicMock
            return mod

        def exec_module(self, module: object) -> None:
            pass

    sys.meta_path.insert(0, HeavyFinder())


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Ensure heavy tests skip gracefully if torch is absent and heavy test is run."""
    if "heavy" in item.keywords and not TORCH_AVAILABLE:
        pytest.skip("Test marked heavy requires real PyTorch installation.")
