"""Backwards-compatibility shim for ParallaxToolDispatcher.

DEPRECATION NOTICE:
    `src.engine.grpc_boundary` has been superseded by `src.engine.dispatcher`.
    Please import `ParallaxToolDispatcher` directly from `src.engine.dispatcher`.
"""

from __future__ import annotations

import warnings

from src.engine.dispatcher import ParallaxToolDispatcher

warnings.warn(
    "src.engine.grpc_boundary is deprecated. Use src.engine.dispatcher instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ParallaxToolDispatcher"]
