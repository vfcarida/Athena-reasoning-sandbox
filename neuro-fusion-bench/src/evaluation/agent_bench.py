"""Agent-Bench Suite — Deterministic evaluation metrics for AI agent traces.

This module implements quantitative evaluation of AI agent behavior by comparing
the agent's actual tool usage and plan execution against ground truth references.

Metrics implemented:
    - **Tool Correctness**: Precision of tool calls against the expected sequence.
    - **Tool Efficiency**: Ratio of optimal to actual tool calls (penalizes redundancy).
    - **Redundancy Rate**: Fraction of duplicate/unnecessary tool calls.
    - **Plan Adherence**: Sequential alignment between planned and executed steps
      using Longest Common Subsequence (LCS) analysis.

All metrics are deterministic (no LLM calls required) and return scores in [0, 1].

References:
    - Liu et al. (2023). AgentBench: Evaluating LLMs as Agents.
    - Mialon et al. (2023). GAIA: A Benchmark for General AI Assistants.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolEvaluationResult:
    """Container for tool usage evaluation metrics.

    Attributes:
        tool_correctness: Fraction of expected tools that were correctly invoked.
            Range: [0.0, 1.0]. Higher is better.
        tool_efficiency: Ratio of minimum required calls to actual calls.
            Range: [0.0, 1.0]. 1.0 means perfectly efficient.
        redundancy_rate: Fraction of tool calls that were redundant (duplicates
            beyond what is required). Range: [0.0, 1.0]. Lower is better.
        parameter_accuracy: Fraction of tool calls with correct parameters.
            Range: [0.0, 1.0]. Higher is better.
        details: Granular breakdown of each tool call's evaluation.
    """
    tool_correctness: float
    tool_efficiency: float
    redundancy_rate: float
    parameter_accuracy: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a flat dictionary for reporting."""
        return {
            "tool_correctness": round(self.tool_correctness, 4),
            "tool_efficiency": round(self.tool_efficiency, 4),
            "redundancy_rate": round(self.redundancy_rate, 4),
            "parameter_accuracy": round(self.parameter_accuracy, 4),
            "details": self.details,
        }


@dataclass
class PlanAdherenceResult:
    """Container for plan adherence evaluation.

    Attributes:
        adherence_score: Normalized LCS ratio measuring sequential alignment.
            Range: [0.0, 1.0]. Higher means better plan following.
        lcs_length: Length of the Longest Common Subsequence.
        plan_length: Length of the reference plan.
        actual_length: Length of the actual execution trace.
        matched_steps: List of steps that matched between plan and execution.
        missed_steps: Plan steps that were not executed.
        extra_steps: Executed steps not in the original plan.
    """
    adherence_score: float
    lcs_length: int
    plan_length: int
    actual_length: int
    matched_steps: list[str] = field(default_factory=list)
    missed_steps: list[str] = field(default_factory=list)
    extra_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a flat dictionary for reporting."""
        return {
            "adherence_score": round(self.adherence_score, 4),
            "lcs_length": self.lcs_length,
            "plan_length": self.plan_length,
            "actual_length": self.actual_length,
            "matched_steps": self.matched_steps,
            "missed_steps": self.missed_steps,
            "extra_steps": self.extra_steps,
        }


class AgentBenchSuite:
    """Deterministic evaluation framework for AI agent tool usage and planning.

    Provides quantitative metrics for assessing agent behavior by comparing
    actual execution traces against ground truth references. All computations
    are fully deterministic — no LLM calls are required.

    Example:
        >>> suite = AgentBenchSuite()
        >>> trace = [
        ...     {"tool": "search", "params": {"query": "weather"}},
        ...     {"tool": "click", "params": {"element": "result_1"}},
        ...     {"tool": "search", "params": {"query": "weather"}},  # redundant
        ... ]
        >>> ground_truth = ["search", "click"]
        >>> result = suite.evaluate_tool_usage(trace, ground_truth)
        >>> print(f"Correctness: {result.tool_correctness}")
    """

    def __init__(self) -> None:
        """Initialize the Agent-Bench evaluation suite."""
        logger.info("AgentBenchSuite initialized.")

    def evaluate_tool_usage(
        self,
        agent_trace: list[dict[str, Any]],
        ground_truth_tools: list[str],
        expected_params: Optional[list[dict[str, Any]]] = None,
    ) -> ToolEvaluationResult:
        """Evaluate tool call correctness and efficiency against a ground truth.

        Computes four metrics:

        1. **Tool Correctness** (Recall): What fraction of expected tools did the
           agent actually invoke?

               Correctness = |{expected} ∩ {actual_unique}| / |{expected}|

        2. **Tool Efficiency**: How close was the agent to the optimal number of
           tool calls?

               Efficiency = |expected_sequence| / max(|actual_calls|, 1)

        3. **Redundancy Rate**: What fraction of calls were duplicates beyond the
           expected count?

               Redundancy = redundant_calls / max(total_calls, 1)

        4. **Parameter Accuracy**: If expected parameters are provided, what
           fraction of tool calls had correct parameters?

        Args:
            agent_trace: List of dictionaries, each representing a tool call.
                Required key: ``"tool"`` (str). Optional key: ``"params"`` (dict).
            ground_truth_tools: Ordered list of expected tool names for the task.
            expected_params: Optional list of expected parameter dictionaries,
                positionally aligned with ``ground_truth_tools``. If None,
                parameter accuracy is reported as 1.0.

        Returns:
            :class:`ToolEvaluationResult` with all computed metrics.

        Raises:
            ValueError: If ``agent_trace`` contains entries without a ``"tool"`` key.
        """
        if not ground_truth_tools:
            return ToolEvaluationResult(
                tool_correctness=1.0,
                tool_efficiency=1.0,
                redundancy_rate=0.0,
                parameter_accuracy=1.0,
                details={"note": "No ground truth tools specified."},
            )

        # Validate trace entries
        for i, entry in enumerate(agent_trace):
            if "tool" not in entry:
                raise ValueError(
                    f"Trace entry {i} missing 'tool' key: {entry}"
                )

        # Extract tool names from agent trace
        actual_tools = [entry["tool"] for entry in agent_trace]
        actual_count = len(actual_tools)

        # ─── Tool Correctness (Recall) ───────────────────────────────
        expected_set = set(ground_truth_tools)
        actual_unique = set(actual_tools)
        correct_tools = expected_set & actual_unique
        tool_correctness = len(correct_tools) / len(expected_set)

        # ─── Tool Efficiency ─────────────────────────────────────────
        expected_count = len(ground_truth_tools)
        tool_efficiency = min(expected_count / max(actual_count, 1), 1.0)

        # ─── Redundancy Rate ─────────────────────────────────────────
        # Count expected frequency of each tool
        expected_counter = Counter(ground_truth_tools)
        actual_counter = Counter(actual_tools)

        redundant_calls = 0
        for tool, actual_freq in actual_counter.items():
            expected_freq = expected_counter.get(tool, 0)
            if actual_freq > expected_freq:
                redundant_calls += actual_freq - expected_freq

        redundancy_rate = redundant_calls / max(actual_count, 1)

        # ─── Parameter Accuracy ──────────────────────────────────────
        parameter_accuracy = 1.0
        if expected_params is not None and agent_trace:
            correct_params = 0
            total_checked = 0

            for i, expected in enumerate(expected_params):
                if i >= len(agent_trace):
                    break
                actual_params = agent_trace[i].get("params", {})
                total_checked += 1
                if actual_params == expected:
                    correct_params += 1

            parameter_accuracy = correct_params / max(total_checked, 1)

        # Build details
        details = {
            "expected_tools": ground_truth_tools,
            "actual_tools": actual_tools,
            "correct_tools": sorted(correct_tools),
            "missing_tools": sorted(expected_set - actual_unique),
            "unexpected_tools": sorted(actual_unique - expected_set),
            "redundant_calls": redundant_calls,
            "total_actual_calls": actual_count,
            "total_expected_calls": expected_count,
        }

        logger.info(
            "Tool evaluation: correctness=%.4f, efficiency=%.4f, redundancy=%.4f",
            tool_correctness, tool_efficiency, redundancy_rate,
        )

        return ToolEvaluationResult(
            tool_correctness=tool_correctness,
            tool_efficiency=tool_efficiency,
            redundancy_rate=redundancy_rate,
            parameter_accuracy=parameter_accuracy,
            details=details,
        )

    def evaluate_plan_adherence(
        self,
        plan: list[str],
        actual_steps: list[str],
    ) -> PlanAdherenceResult:
        """Evaluate sequential adherence of execution trace to a planned sequence.

        Uses the **Longest Common Subsequence (LCS)** algorithm to measure how
        well the agent followed its own plan. The adherence score is normalized
        by the maximum of plan and actual lengths:

            Adherence = len(LCS(plan, actual)) / max(len(plan), len(actual))

        This metric captures:
        - **Step omission**: Plan steps that were skipped (penalizes adherence).
        - **Step insertion**: Extra steps not in the plan (penalizes adherence).
        - **Step reordering**: Steps executed out of planned order.

        Args:
            plan: Ordered list of planned step names/descriptions.
            actual_steps: Ordered list of actually executed step names.

        Returns:
            :class:`PlanAdherenceResult` with adherence score and alignment details.
        """
        if not plan and not actual_steps:
            return PlanAdherenceResult(
                adherence_score=1.0,
                lcs_length=0,
                plan_length=0,
                actual_length=0,
            )

        if not plan or not actual_steps:
            return PlanAdherenceResult(
                adherence_score=0.0,
                lcs_length=0,
                plan_length=len(plan),
                actual_length=len(actual_steps),
                missed_steps=list(plan),
                extra_steps=list(actual_steps),
            )

        # ─── LCS Dynamic Programming ────────────────────────────────
        n, m = len(plan), len(actual_steps)
        dp = [[0] * (m + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if plan[i - 1] == actual_steps[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_length = dp[n][m]

        # ─── Backtrack to find the matched steps ─────────────────────
        matched_steps: list[str] = []
        i, j = n, m
        while i > 0 and j > 0:
            if plan[i - 1] == actual_steps[j - 1]:
                matched_steps.append(plan[i - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] > dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
        matched_steps.reverse()

        # ─── Identify missed and extra steps ─────────────────────────
        matched_set_plan = set()
        matched_set_actual = set()
        plan_copy = list(plan)
        actual_copy = list(actual_steps)

        # Track which indices in plan/actual were matched
        pi, ai = 0, 0
        matched_plan_indices: set[int] = set()
        matched_actual_indices: set[int] = set()

        for step in matched_steps:
            while pi < n and (pi in matched_plan_indices or plan[pi] != step):
                pi += 1
            while ai < m and (ai in matched_actual_indices or actual_steps[ai] != step):
                ai += 1
            if pi < n and ai < m:
                matched_plan_indices.add(pi)
                matched_actual_indices.add(ai)
                pi += 1
                ai += 1

        missed_steps = [plan[i] for i in range(n) if i not in matched_plan_indices]
        extra_steps = [actual_steps[i] for i in range(m) if i not in matched_actual_indices]

        # Adherence score
        max_len = max(n, m)
        adherence_score = lcs_length / max_len if max_len > 0 else 1.0

        logger.info(
            "Plan adherence: score=%.4f, LCS=%d/%d",
            adherence_score, lcs_length, max_len,
        )

        return PlanAdherenceResult(
            adherence_score=adherence_score,
            lcs_length=lcs_length,
            plan_length=n,
            actual_length=m,
            matched_steps=matched_steps,
            missed_steps=missed_steps,
            extra_steps=extra_steps,
        )

    def full_evaluation(
        self,
        agent_trace: list[dict[str, Any]],
        ground_truth_tools: list[str],
        plan: list[str],
        actual_steps: list[str],
        expected_params: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Run a complete evaluation combining tool usage and plan adherence.

        Convenience method that runs both :meth:`evaluate_tool_usage` and
        :meth:`evaluate_plan_adherence`, then combines results into a single
        report dictionary.

        Args:
            agent_trace: Tool call trace from the agent.
            ground_truth_tools: Expected tool sequence.
            plan: The agent's declared plan.
            actual_steps: The agent's actual execution steps.
            expected_params: Optional expected parameters for tool calls.

        Returns:
            Combined evaluation dictionary with all metrics.
        """
        tool_result = self.evaluate_tool_usage(
            agent_trace, ground_truth_tools, expected_params
        )
        plan_result = self.evaluate_plan_adherence(plan, actual_steps)

        return {
            "tool_evaluation": tool_result.to_dict(),
            "plan_evaluation": plan_result.to_dict(),
            "composite_score": round(
                (tool_result.tool_correctness * 0.3
                 + tool_result.tool_efficiency * 0.2
                 + (1.0 - tool_result.redundancy_rate) * 0.1
                 + plan_result.adherence_score * 0.4),
                4,
            ),
        }
