"""Executive Engine module implementing Cognitive-Executive Separation (Parallax Principle)."""

from src.engine.executor import ExecutiveEngineProcess
from src.engine.grpc_boundary import ParallaxToolDispatcher

__all__ = ["ExecutiveEngineProcess", "ParallaxToolDispatcher"]
