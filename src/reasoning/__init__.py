"""Reasoning subpackage — Entropy-guided adaptive inference engine."""

from .schemas import (
    ActionType,
    AgentPlan,
    ObservationPayload,
    ReasoningStep,
    ToolCallPayload,
)

try:
    from .swi_reasoning import SwiReasoningEngine, SwiReasoningSimulator
except ImportError:
    SwiReasoningEngine = None  # type: ignore[assignment, misc]
    SwiReasoningSimulator = None  # type: ignore[assignment, misc]

__all__ = [
    "ActionType",
    "AgentPlan",
    "ObservationPayload",
    "ReasoningStep",
    "ToolCallPayload",
    "SwiReasoningEngine",
    "SwiReasoningSimulator",
]
