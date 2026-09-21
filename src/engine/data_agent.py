"""Autonomous FinOps Data Lake Agent with Multi-Turn Adaptive Reflection.

Connects entropy-guided reasoning dynamics to the Executive Agent Loop.
Formulates structured SQL queries, evaluates them against pre-execution AST guards,
and autonomously reflects upon FinOps rejections (partition omission, unbounded projections)
to produce self-correcting query execution trajectories.

Algorithmic Complexity:
    - Multi-Turn Reflection: O(R * (P + E)) where R is reflection turns, P is planning latency,
      and E is AST validation and execution time.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.athena.athena_client import AthenaClient
from src.athena.duckdb_client import DuckDBClient
from src.engine.agent_loop import AgentLoop, AgentPlanner
from src.engine.dispatcher import ParallaxToolDispatcher
from src.engine.executor import ExecutiveEngineProcess
from src.reasoning.schemas import ActionType, ReasoningStep
from src.state.checkpoint_manager import (
    ConversationCheckpointManager,
    ConversationStateCheckpoint,
)

logger = logging.getLogger(__name__)


class DataAgentExecutionSummary(BaseModel):
    """Execution summary produced by the AutonomousDataAgent after multi-turn execution.

    Attributes:
        goal: The initial user goal or analytics objective.
        success: True if query was validated and executed successfully.
        turns: Total number of execution and reflection attempts.
        thinking_log: Sequential list of reasoning and reflection chains.
        final_sql: The final compliant SQL statement executed.
        result_data: Query output rows and execution metadata.
        error: Terminal error message if all turns failed.
        total_time_ms: Total elapsed execution time in milliseconds.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    goal: str = Field(description="The user's original analytics goal.")
    success: bool = Field(description="Whether the task succeeded.")
    turns: int = Field(ge=1, description="Total execution turns taken.")
    thinking_log: list[str] = Field(
        default_factory=list,
        description="Sequential list of reasoning thoughts and reflections.",
    )
    final_sql: str | None = Field(default=None, description="Final executed SQL string.")
    result_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured output data from the successful query.",
    )
    error: str | None = Field(default=None, description="Terminal error description if failed.")
    total_time_ms: float = Field(ge=0.0, description="Total execution time in milliseconds.")


class AutonomousDataAgent:
    """Autonomous reasoning agent for FinOps-governed data lake analytics.

    Coordinates cognitive reasoning, AST query generation, and multi-turn self-correction
    when queries violate FinOps partition pruning or unbounded projection rules.
    """

    def __init__(
        self,
        backend: str = "duckdb",
        duckdb_client: DuckDBClient | None = None,
        athena_client: AthenaClient | None = None,
        checkpoint_manager: ConversationCheckpointManager | None = None,
        max_reflection_turns: int = 3,
    ) -> None:
        """Initialize the autonomous data agent.

        Args:
            backend: Target execution engine ("duckdb" or "athena").
            duckdb_client: Optional custom DuckDBClient instance.
            athena_client: Optional custom AthenaClient instance.
            checkpoint_manager: Optional conversation checkpoint manager for state persistence.
            max_reflection_turns: Maximum allowed correction cycles upon query failure.
        """
        self.backend = backend
        self.duckdb_client = duckdb_client or (DuckDBClient() if backend == "duckdb" else None)
        self.athena_client = athena_client or (AthenaClient() if backend == "athena" else None)
        self.checkpoint_manager = checkpoint_manager
        self.max_reflection_turns = max_reflection_turns
        self.planner = AgentPlanner()

        # Wire executive engine and Parallax dispatcher
        self.executor = ExecutiveEngineProcess(
            athena_client=self.athena_client,
            duckdb_client=self.duckdb_client,
        )
        self.dispatcher = ParallaxToolDispatcher(executor=self.executor)
        self.agent_loop = AgentLoop(dispatcher=self.dispatcher)

    def _reflect_and_refine_sql(
        self,
        original_sql: str,
        error_message: str,
        default_partition_key: str = "dt",
        default_partition_val: str = "2026-09-01",
    ) -> tuple[str, str]:
        """Reflect upon a FinOps rejection and synthesize a compliant SQL statement.

        Returns:
            Tuple of (refined_sql, reasoning_reflection_thought).
        """
        refined = original_sql.strip().rstrip(";").strip()

        # 1. Reflection on missing partition filter
        if "WHERE clause" in error_message or "partition filter" in error_message:
            thought = (
                f"<think> Reflection: The query was rejected because it omits partition pruning "
                f"on column '{default_partition_key}'. Scans on analytical data lakes require "
                f"explicit partition bounding to avoid full table scans ($5/TB risk). "
                f"Appending partition filter: WHERE {default_partition_key} = '{default_partition_val}'. </think>"
            )
            if re.search(r"\bWHERE\b", refined, flags=re.IGNORECASE):
                refined = re.sub(
                    r"\bWHERE\b",
                    f"WHERE {default_partition_key} = '{default_partition_val}' AND",
                    refined,
                    count=1,
                    flags=re.IGNORECASE,
                )
            elif re.search(r"\bLIMIT\b", refined, flags=re.IGNORECASE):
                refined = re.sub(
                    r"\bLIMIT\b",
                    f"WHERE {default_partition_key} = '{default_partition_val}' LIMIT",
                    refined,
                    count=1,
                    flags=re.IGNORECASE,
                )
            else:
                refined = f"{refined} WHERE {default_partition_key} = '{default_partition_val}'"

        # 2. Reflection on unbounded SELECT *
        elif "Unbounded 'SELECT *" in error_message or "Unbounded projection" in error_message:
            thought = (
                "<think> Reflection: The query was rejected due to an unbounded 'SELECT *' projection. "
                "Columnar storage engines scan every column block in SELECT *. "
                "Restricting projection to explicit columns: 'order_id, amount'. </think>"
            )
            refined = re.sub(r"SELECT\s+\*", "SELECT order_id, amount", refined, count=1, flags=re.IGNORECASE)

        # 3. Reflection on missing LIMIT
        elif "LIMIT" in error_message:
            thought = (
                "<think> Reflection: Query lacks an explicit LIMIT clause. "
                "Appending 'LIMIT 100' to bound client result buffer. </think>"
            )
            refined = f"{refined} LIMIT 100"

        else:
            thought = f"<think> Reflection: Query failed with error '{error_message}'. Applying default hygiene: adding LIMIT 100. </think>"
            if not re.search(r"\bLIMIT\b", refined, flags=re.IGNORECASE):
                refined = f"{refined} LIMIT 100"

        return refined, thought

    async def run(
        self,
        goal: str,
        initial_sql: str,
        required_partition_keys: list[str] | None = None,
    ) -> DataAgentExecutionSummary:
        """Execute the analytical objective with autonomous multi-turn reflection.

        Args:
            goal: Natural language goal description.
            initial_sql: Initial proposed SQL query string.
            required_partition_keys: Partition key names required in query filter.

        Returns:
            DataAgentExecutionSummary detailing execution results and reflections.
        """
        start_time = time.perf_counter()
        tool_name = "duckdb_query" if self.backend == "duckdb" else "athena_query"
        action_type = ActionType.ATHENA_QUERY if self.backend == "athena" else ActionType.ATHENA_QUERY

        current_sql = initial_sql
        thinking_log: list[str] = [
            f"<think> Initial Formulation: Proposed SQL query for objective '{goal}': {current_sql} </think>"
        ]
        last_error: str | None = None
        final_output: dict[str, Any] = {}

        for turn in range(1, self.max_reflection_turns + 1):
            logger.info("Data Agent execution turn %d/%d (SQL: %s)", turn, self.max_reflection_turns, current_sql[:60])

            # Build single-step query plan
            step = ReasoningStep(
                step_number=1,
                rationale=f"Execute query on {self.backend} data lake backend.",
                action_type=action_type,
                tool_name=tool_name,
                tool_args={
                    "sql": current_sql,
                    "required_partition_keys": required_partition_keys,
                    "enforce_limit": True,
                },
            )
            plan = self.planner.create_plan(
                task_goal=goal,
                steps=[step],
                plan_id=str(uuid.uuid4()),
                thinking_process=thinking_log[-1],
                estimated_complexity=2,
            )

            # Execute plan across Parallax boundary
            checkpoint = (
                ConversationStateCheckpoint(checkpoint_id=str(uuid.uuid4()), step_index=0)
                if self.checkpoint_manager
                else None
            )
            loop_result = await self.agent_loop.run_plan(
                plan=plan,
                checkpoint=checkpoint,
                checkpoint_manager=self.checkpoint_manager,
            )

            if loop_result.success:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                final_output = loop_result.final_output or {}
                thinking_log.append(f"<think> Success: Query executed successfully on turn {turn}. </think>")
                return DataAgentExecutionSummary(
                    goal=goal,
                    success=True,
                    turns=turn,
                    thinking_log=thinking_log,
                    final_sql=current_sql,
                    result_data=final_output,
                    error=None,
                    total_time_ms=elapsed_ms,
                )

            # Query failed (e.g. FinOps AST rejection)
            last_error = loop_result.error or "Unknown execution error"
            logger.warning("Turn %d failed with error: %s", turn, last_error)

            if turn < self.max_reflection_turns:
                # Enter reflection mode and refine query
                refined_sql, reflection_thought = self._reflect_and_refine_sql(
                    original_sql=current_sql,
                    error_message=last_error,
                    default_partition_key=(required_partition_keys[0] if required_partition_keys else "dt"),
                )
                thinking_log.append(reflection_thought)
                current_sql = refined_sql

        # All reflection turns exhausted
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return DataAgentExecutionSummary(
            goal=goal,
            success=False,
            turns=self.max_reflection_turns,
            thinking_log=thinking_log,
            final_sql=current_sql,
            result_data={},
            error=last_error,
            total_time_ms=elapsed_ms,
        )


__all__ = ["AutonomousDataAgent", "DataAgentExecutionSummary"]
