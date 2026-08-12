"""Type-safe Pydantic output schemas for Plan-then-Execute (P-t-E) agentic reasoning.

This module defines strict structured output contracts for LLM plans, tool invocations,
and executive observations. It enforces validation before any tool dispatch occurs,
preventing malformed or non-deterministic payload execution.

Algorithmic Complexity:
    - Validation: O(N) where N is the depth/size of the JSON payload.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class ActionType(str, Enum):
    """Supported executive action types."""
    ATHENA_QUERY = "athena_query"
    SANDBOX_EXECUTE = "sandbox_execute"
    RETRIEVAL_SEARCH = "retrieval_search"
    STATE_CHECKPOINT = "state_checkpoint"
    FINAL_ANSWER = "final_answer"


class ReasoningStep(BaseModel):
    """Represents a discrete step within an agent's reasoning plan.

    Attributes:
        step_number: Monotonically increasing step index (1-indexed).
        rationale: Natural language explanation of why this step is necessary.
        action_type: Category of action to execute.
        tool_name: Name of target tool handler to invoke.
        tool_args: Dictionary of JSON arguments for the tool handler.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    step_number: int = Field(ge=1, description="Sequential step index.")
    rationale: str = Field(min_length=5, description="Logical justification for this step.")
    action_type: ActionType = Field(description="Target action classification.")
    tool_name: str = Field(description="Identifier of target tool handler.")
    tool_args: Dict[str, Any] = Field(default_factory=dict, description="Arguments payload.")


class AgentPlan(BaseModel):
    """Structured plan emitted by Cognitive Reasoning Layer before execution.

    Enforces the Plan-then-Execute (P-t-E) paradigm by requiring a complete,
    validated trajectory of steps prior to executive invocation.

    Attributes:
        plan_id: Unique identifier for the generated reasoning trajectory.
        task_goal: Overall goal or user prompt being addressed.
        thinking_process: Content generated during latent thinking phases (<think>...</think>).
        steps: List of validated ReasoningStep objects.
        estimated_complexity: Qualitative difficulty score (1-10).
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str = Field(description="Unique UUID string identifying the plan.")
    task_goal: str = Field(min_length=3, description="Target task objective.")
    thinking_process: Optional[str] = Field(default=None, description="Latent reasoning trace.")
    steps: List[ReasoningStep] = Field(min_items=1, description="Ordered list of execution steps.")
    estimated_complexity: int = Field(default=1, ge=1, le=10, description="Estimated complexity score.")


class ToolCallPayload(BaseModel):
    """Encapsulates a single validated tool dispatch request across the Parallax boundary.

    Attributes:
        call_id: Unique invocation handle.
        tool_name: Target tool name.
        arguments: Validated JSON arguments.
        timeout_seconds: Hard timeout limit for execution.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    call_id: str = Field(description="Unique call tracking ID.")
    tool_name: str = Field(description="Registered tool name.")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool parameter payload.")
    timeout_seconds: float = Field(default=30.0, gt=0.0, description="Execution timeout.")


class ObservationPayload(BaseModel):
    """Structured result returned across the Parallax boundary from Executive Layer.

    Attributes:
        call_id: Matching call tracking ID.
        success: Boolean flag indicating clean execution without runtime errors.
        output_data: Structured output object returned by tool handler.
        error_message: Error description if execution failed.
        execution_time_ms: Total latency in milliseconds.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    call_id: str = Field(description="Matching invocation identifier.")
    success: bool = Field(description="Execution success status.")
    output_data: Dict[str, Any] = Field(default_factory=dict, description="Output payload dictionary.")
    error_message: Optional[str] = Field(default=None, description="Detailed error trace if failed.")
    execution_time_ms: float = Field(ge=0.0, description="Execution runtime in milliseconds.")
