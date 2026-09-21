"""Agent Loop implementation for Plan-then-Execute (P-t-E) orchestration.

Coordinates the reasoning, planning, and execution phases across the
Parallax security boundary. Supports resilient execution with conversation-level
checkpoints and idempotent effect ledgers.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.engine.dispatcher import ParallaxToolDispatcher
from src.engine.executor import ExecutiveEngineProcess
from src.reasoning.schemas import (
    ActionType,
    AgentPlan,
    ObservationPayload,
    ReasoningStep,
    ToolCallPayload,
)
from src.state.checkpoint_manager import (
    ConversationCheckpointManager,
    ConversationStateCheckpoint,
    compute_idempotency_key,
)

logger = logging.getLogger(__name__)


class AgentLoopResult(BaseModel):
    """Structured result returned by the AgentLoop after executing an AgentPlan.

    Attributes:
        plan_id: Identifier of the executed plan.
        success: True if all steps in the plan executed successfully.
        step_results: List of ObservationPayload objects for executed steps.
        final_output: Output data dictionary of the terminal step, if successful.
        error: Error message from the failed step, if any.
        total_execution_time_ms: Total duration of execution in milliseconds.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str = Field(description="Identifier of the executed plan.")
    success: bool = Field(description="Whether the entire plan executed successfully.")
    step_results: list[ObservationPayload] = Field(
        default_factory=list,
        description="Sequential list of observation results from executed steps.",
    )
    final_output: dict[str, Any] | None = Field(
        default=None,
        description="Output of final step on success.",
    )
    error: str | None = Field(
        default=None,
        description="Error description if any step failed.",
    )
    total_execution_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total duration of execution in milliseconds.",
    )


class AgentPlanner:
    """Planner for creating validated, immutable AgentPlan instances."""

    @staticmethod
    def estimate_complexity(task_goal: str, step_count: int = 1) -> int:
        """Heuristic estimator for plan complexity score (1-10).

        Args:
            task_goal: The description of the task.
            step_count: Number of steps planned.

        Returns:
            Integer complexity score between 1 and 10.
        """
        score = min(max(step_count * 2, 1), 10)
        keywords = ("optimize", "benchmark", "sandbox", "athena", "pipeline", "distributed")
        matches = sum(1 for kw in keywords if kw in task_goal.lower())
        score = min(score + matches, 10)
        return max(1, min(score, 10))

    def create_plan(
        self,
        task_goal: str,
        steps: list[ReasoningStep],
        plan_id: str | None = None,
        thinking_process: str | None = None,
        estimated_complexity: int | None = None,
    ) -> AgentPlan:
        """Create and validate a frozen AgentPlan.

        Args:
            task_goal: Overall goal of the agent.
            steps: List of ordered reasoning/execution steps.
            plan_id: Optional unique identifier (generated if omitted).
            thinking_process: Optional chain-of-thought or reasoning notes.
            estimated_complexity: Optional complexity (computed if omitted).

        Returns:
            Validated, immutable AgentPlan instance.
        """
        pid = plan_id or str(uuid.uuid4())
        complexity = (
            estimated_complexity
            if estimated_complexity is not None
            else self.estimate_complexity(task_goal, len(steps))
        )
        return AgentPlan(
            plan_id=pid,
            task_goal=task_goal,
            thinking_process=thinking_process,
            steps=steps,
            estimated_complexity=complexity,
        )

    @staticmethod
    def create_athena_step(
        step_number: int,
        rationale: str,
        sql: str,
        required_partition_keys: list[str] | None = None,
        enforce_limit: bool = True,
        dry_run: bool = False,
        wait: bool = False,
        timeout_seconds: float = 30.0,
    ) -> ReasoningStep:
        """Helper to construct an Athena query reasoning step."""
        args: dict[str, Any] = {
            "sql": sql,
            "enforce_limit": enforce_limit,
            "dry_run": dry_run,
            "wait": wait,
            "timeout_seconds": timeout_seconds,
        }
        if required_partition_keys is not None:
            args["required_partition_keys"] = required_partition_keys

        return ReasoningStep(
            step_number=step_number,
            rationale=rationale,
            action_type=ActionType.ATHENA_QUERY,
            tool_name="athena_query",
            tool_args=args,
        )

    @staticmethod
    def create_sandbox_step(
        step_number: int,
        rationale: str,
        code: str,
        language: str = "python",
        environment_vars: dict[str, str] | None = None,
        timeout_seconds: float = 30.0,
    ) -> ReasoningStep:
        """Helper to construct a Sandbox execution reasoning step."""
        args: dict[str, Any] = {
            "code": code,
            "language": language,
            "timeout_seconds": timeout_seconds,
        }
        if environment_vars is not None:
            args["environment_vars"] = environment_vars

        return ReasoningStep(
            step_number=step_number,
            rationale=rationale,
            action_type=ActionType.SANDBOX_EXECUTE,
            tool_name="sandbox_execute",
            tool_args=args,
        )


class AgentLoop:
    """Coordinates execution of an AgentPlan via the Parallax boundary.

    Enforces that steps are dispatched in order, respects timeouts and authorization,
    guards against repeated side effects using an effect ledger, and propagates failures
    immediately with typed results.
    """

    def __init__(
        self,
        dispatcher: ParallaxToolDispatcher | None = None,
        executor: ExecutiveEngineProcess | None = None,
        planner: AgentPlanner | None = None,
    ) -> None:
        """Initialize the AgentLoop.

        Args:
            dispatcher: Optional custom ParallaxToolDispatcher.
            executor: Optional custom ExecutiveEngineProcess (used if dispatcher is None).
            planner: Optional AgentPlanner instance.
        """
        if dispatcher is not None:
            self.dispatcher = dispatcher
        elif executor is not None:
            self.dispatcher = ParallaxToolDispatcher(executor=executor)
        else:
            self.dispatcher = ParallaxToolDispatcher()

        self.planner = planner or AgentPlanner()

    async def run_plan(
        self,
        plan: AgentPlan,
        checkpoint: ConversationStateCheckpoint | None = None,
        checkpoint_manager: ConversationCheckpointManager | None = None,
    ) -> AgentLoopResult:
        """Execute each step of the provided plan sequentially.

        If a checkpoint is provided, steps whose side effects have already been
        recorded in the effect ledger will be skipped, returning their cached output.

        Args:
            plan: Validated AgentPlan.
            checkpoint: Optional ConversationStateCheckpoint tracking state and effects.
            checkpoint_manager: Optional manager to persist checkpoint progress.

        Returns:
            AgentLoopResult containing the final status and all step observations.
        """
        start_time = time.perf_counter()
        step_results: list[ObservationPayload] = []

        logger.info("Starting execution of plan '%s' with %d steps", plan.plan_id, len(plan.steps))

        for step in plan.steps:
            action_type_str = (
                step.action_type.value
                if hasattr(step.action_type, "value")
                else str(step.action_type)
            )
            idempotency_key = compute_idempotency_key(
                action_type_str, step.tool_name, step.tool_args
            )

            # Check if this effect has already been recorded
            if checkpoint is not None and checkpoint.has_effect(idempotency_key):
                logger.info(
                    "Step %d (%s) already executed. Replaying from effect ledger.",
                    step.step_number,
                    step.tool_name,
                )
                cached_data = checkpoint.get_effect(idempotency_key) or {}
                cached_obs = ObservationPayload(
                    call_id=f"{plan.plan_id}-step-{step.step_number}",
                    success=True,
                    output_data=cached_data,
                    error_message=None,
                    execution_time_ms=0.0,
                )
                step_results.append(cached_obs)
                continue

            timeout = float(step.tool_args.get("timeout_seconds", 30.0))
            call_id = f"{plan.plan_id}-step-{step.step_number}"

            payload = ToolCallPayload(
                call_id=call_id,
                tool_name=step.tool_name,
                arguments=step.tool_args,
                timeout_seconds=timeout,
            )

            observation = await self.dispatcher.dispatch_tool_call(payload)
            step_results.append(observation)

            if not observation.success:
                elapsed = (time.perf_counter() - start_time) * 1000.0
                logger.warning(
                    "Plan '%s' halted at step %d (%s) due to error: %s",
                    plan.plan_id,
                    step.step_number,
                    step.tool_name,
                    observation.error_message,
                )
                if checkpoint is not None and checkpoint_manager is not None:
                    checkpoint_manager.save_checkpoint(checkpoint)

                return AgentLoopResult(
                    plan_id=plan.plan_id,
                    success=False,
                    step_results=step_results,
                    final_output=None,
                    error=observation.error_message,
                    total_execution_time_ms=elapsed,
                )

            # Record effect on successful step
            if checkpoint is not None:
                checkpoint.record_effect(idempotency_key, observation.output_data)
                checkpoint.step_index = step.step_number
                checkpoint.turn_history.append({
                    "step": step.step_number,
                    "action_type": action_type_str,
                    "tool_name": step.tool_name,
                    "output_data": observation.output_data,
                })
                if checkpoint_manager is not None:
                    checkpoint_manager.save_checkpoint(checkpoint)

        elapsed = (time.perf_counter() - start_time) * 1000.0
        final_output = step_results[-1].output_data if step_results else {}
        logger.info("Plan '%s' completed successfully in %.2fms", plan.plan_id, elapsed)

        return AgentLoopResult(
            plan_id=plan.plan_id,
            success=True,
            step_results=step_results,
            final_output=final_output,
            error=None,
            total_execution_time_ms=elapsed,
        )

    async def run_task(
        self,
        task_goal: str,
        steps: list[ReasoningStep],
        thinking_process: str | None = None,
        checkpoint: ConversationStateCheckpoint | None = None,
        checkpoint_manager: ConversationCheckpointManager | None = None,
    ) -> AgentLoopResult:
        """Convenience method to create and execute a plan from steps.

        Args:
            task_goal: Description of the goal.
            steps: List of ReasoningStep items.
            thinking_process: Optional rationale/thought process.
            checkpoint: Optional checkpoint for resumption.
            checkpoint_manager: Optional checkpoint manager for persistence.

        Returns:
            AgentLoopResult
        """
        plan = self.planner.create_plan(
            task_goal=task_goal,
            steps=steps,
            thinking_process=thinking_process,
        )
        return await self.run_plan(plan, checkpoint=checkpoint, checkpoint_manager=checkpoint_manager)
