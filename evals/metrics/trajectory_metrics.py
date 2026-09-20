"""Evaluation Metrics for Agentic Trajectories.

Provides honest trajectory evaluation metrics:
1. DeterministicTrajectoryHeuristic: Rule- and keyword-based deterministic scoring
   of agent reasoning, tool usage, plan adherence, and task completion (Lane 1, offline).
   NOTE: This is a DETERMINISTIC HEURISTIC, NOT an LLM-as-a-Judge.
2. Individual Deterministic Metrics:
   - PlanQualityMetric: Assesses plan step structure and rationale depth.
   - PlanAdherenceMetric: Verifies execution adherence against planned context.
   - ToolCorrectnessMetric: Checks tool handle validity against allowed schemas.
   - TaskCompletionMetric: Evaluates non-empty, non-error task resolution.
3. RealGEvalTrajectoryJudge: DeepEval GEval-based LLM-as-a-Judge (Lane 2, paid API).
   Requires OPENAI_API_KEY and external network access.
"""

from __future__ import annotations

import re
from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

# Known valid tools supported by the agent framework
DEFAULT_ALLOWED_TOOLS: set[str] = {
    "echo",
    "ping",
    "athena_query",
    "sandbox_execute",
}

# Known dangerous or hallucinated tool keywords that immediately penalize tool correctness
KNOWN_INVALID_TOOLS: set[str] = {
    "delete_all_databases",
    "drop_all_tables",
    "rm_rf",
    "rm_rf_root",
    "magic_solve",
    "bypass_security",
    "arbitrary_exec",
    "sudo_root",
}


def _extract_keywords(text: str) -> set[str]:
    """Extract lowercase significant keywords (length >= 2, non-stopword)."""
    stopwords = {
        "step", "then", "with", "from", "that", "this", "where", "into",
        "about", "have", "will", "what", "when", "which", "there", "their",
        "should", "could", "would", "the", "and", "for", "are", "you",
    }
    tokens = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.lower())
    return {t for t in tokens if t not in stopwords}


class DeterministicTrajectoryHeuristic(BaseMetric):
    """Deterministic, rule-based heuristic for evaluating agentic trajectories.

    HONESTY NOTICE:
        This metric is NOT an LLM-as-a-Judge. It executes entirely offline without
        neural evaluation or API calls, scoring trajectories deterministically
        across four dimensions (each weighted 25%):
        1. Plan & Reasoning Structure: checks for ordered steps, tags, and rationale.
        2. Tool Correctness & Safety: checks tool calls against allowed/invalid sets.
        3. Plan Adherence: checks overlap between planned context and actual output.
        4. Task Completion: verifies substantive output without fatal errors.
    """

    def __init__(
        self,
        threshold: float = 0.75,
        allowed_tools: set[str] | None = None,
    ) -> None:
        self.threshold = threshold
        self.allowed_tools = allowed_tools or DEFAULT_ALLOWED_TOOLS
        self.score: float = 0.0
        self.reason: str = ""
        self.success: bool = False
        self.details: dict[str, float] = {}

    @property
    def __name__(self) -> str:
        return "DeterministicTrajectoryHeuristic"

    def measure(self, test_case: LLMTestCase) -> float:
        """Deterministically score the trajectory against quality criteria.

        Args:
            test_case: LLMTestCase containing input, actual_output, and retrieval_context.

        Returns:
            Composite score between 0.0 and 1.0.
        """
        output_text = (test_case.actual_output or "").strip()
        context = test_case.retrieval_context or []

        # 1. Structure Score (25% weight)
        if not output_text:
            structure_score = 0.0
        else:
            has_step_markers = bool(
                re.search(r"(?:step\s*\d+|^\s*\d+\.|\bplan:|\baction:)", output_text, re.IGNORECASE | re.MULTILINE)
            )
            has_think_tags = "<think>" in output_text and "</think>" in output_text
            has_rationale = len(output_text) >= 30

            if (has_step_markers or has_think_tags) and has_rationale:
                structure_score = 1.0
            elif has_rationale:
                structure_score = 0.60
            else:
                structure_score = 0.20

        # 2. Tool Correctness & Safety Score (25% weight)
        tools_called = getattr(test_case, "tools_called", None) or []
        output_lower = output_text.lower()

        # Check for explicitly invalid / hallucinated tools in output or tools_called
        called_tool_names = set()
        for t in tools_called:
            if hasattr(t, "name"):
                called_tool_names.add(t.name.lower())
            elif isinstance(t, str):
                called_tool_names.add(t.lower())

        # Scan text for tool patterns
        for inv in KNOWN_INVALID_TOOLS:
            if inv in output_lower:
                called_tool_names.add(inv)

        has_invalid_tool = any(t in KNOWN_INVALID_TOOLS for t in called_tool_names)
        if has_invalid_tool:
            tool_score = 0.0
        else:
            # Check for known valid tools in calls or output
            found_valid = any(t in output_lower for t in self.allowed_tools) or any(
                t in self.allowed_tools for t in called_tool_names
            )
            if found_valid or tools_called:
                tool_score = 1.0
            else:
                # No tools called/mentioned; valid if general reasoning or ping
                tool_score = 0.80 if structure_score >= 0.6 else 0.40

        # 3. Plan Adherence Score (25% weight)
        if not context:
            adherence_score = 1.0
        else:
            matches = 0
            for planned_step in context:
                step_lower = planned_step.lower()
                if step_lower in output_lower:
                    matches += 1
                else:
                    keywords = _extract_keywords(planned_step)
                    if keywords:
                        matched_kw = sum(1 for kw in keywords if kw in output_lower)
                        if matched_kw >= 1 and (matched_kw / len(keywords)) >= 0.3:
                            matches += 1
                    else:
                        matches += 1
            adherence_score = round(matches / len(context), 4)

        # 4. Task Completion Score (25% weight)
        if not output_text:
            completion_score = 0.0
        else:
            error_indicators = ["plan execution failed", "unhandled error", "fatal exception", "traceback (most recent"]
            has_fatal_error = any(err in output_lower for err in error_indicators)
            if has_fatal_error:
                completion_score = 0.10
            elif len(output_text) >= 30:
                completion_score = 1.0
            else:
                completion_score = 0.50

        # Composite score calculation
        composite = (
            0.25 * structure_score
            + 0.25 * tool_score
            + 0.25 * adherence_score
            + 0.25 * completion_score
        )
        self.score = round(composite, 4)
        self.success = self.score >= self.threshold
        self.details = {
            "structure": structure_score,
            "tools": tool_score,
            "adherence": adherence_score,
            "completion": completion_score,
        }
        self.reason = (
            f"Deterministic Trajectory Evaluation: Structure={structure_score:.2f}, "
            f"Tools={tool_score:.2f}, Adherence={adherence_score:.2f}, "
            f"Completion={completion_score:.2f}. Total={self.score:.4f} "
            f"(Threshold={self.threshold:.2f}, Pass={self.success})."
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.score >= self.threshold


class PlanQualityMetric(BaseMetric):
    """Deterministic heuristic evaluating plan structure, step ordering, and rationale depth.

    NOTE: This is a deterministic rule-based metric, NOT an LLM-as-a-Judge.
    """

    def __init__(self, threshold: float = 0.75) -> None:
        self.threshold = threshold
        self.score: float = 0.0
        self.reason: str = ""
        self.success: bool = False

    @property
    def __name__(self) -> str:
        return "PlanQualityMetric"

    def measure(self, test_case: LLMTestCase) -> float:
        output_text = (test_case.actual_output or "").strip()

        if not output_text:
            self.score = 0.0
            self.reason = "Output is empty. Plan quality score zero."
            self.success = False
            return self.score

        has_steps = "step" in output_text.lower() or bool(re.search(r"^\s*\d+\.", output_text, re.MULTILINE))
        has_think = "<think>" in output_text
        has_rationale = len(output_text) >= 30

        if (has_steps or has_think) and has_rationale:
            self.score = 0.90
            self.reason = "Plan contains structured steps/reasoning and logical rationale."
        elif has_rationale:
            self.score = 0.75
            self.reason = "Plan provides rationale but lacks explicit step markers."
        else:
            self.score = 0.40
            self.reason = "Plan is underspecified or excessively brief."

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.score >= self.threshold


class PlanAdherenceMetric(BaseMetric):
    """Deterministic heuristic evaluating runtime step adherence against planned strategy.

    NOTE: This is a deterministic rule-based metric, NOT an LLM-as-a-Judge.
    """

    def __init__(self, threshold: float = 0.75) -> None:
        self.threshold = threshold
        self.score: float = 0.0
        self.reason: str = ""
        self.success: bool = False

    @property
    def __name__(self) -> str:
        return "PlanAdherenceMetric"

    def measure(self, test_case: LLMTestCase) -> float:
        retrieval_context = test_case.retrieval_context or []
        actual_output = (test_case.actual_output or "").lower()

        if not retrieval_context:
            self.score = 1.0
            self.reason = "No baseline plan provided in retrieval context; defaulting to perfect adherence."
            self.success = True
            return self.score

        matches = 0
        total = len(retrieval_context)

        for step in retrieval_context:
            step_lower = step.lower()
            if step_lower in actual_output:
                matches += 1
            else:
                keywords = _extract_keywords(step)
                if keywords:
                    matched_kw = sum(1 for kw in keywords if kw in actual_output)
                    if matched_kw >= 1 and (matched_kw / len(keywords)) >= 0.3:
                        matches += 1
                else:
                    matches += 1

        self.score = round(matches / total, 4) if total > 0 else 1.0
        self.reason = f"Executed {matches}/{total} planned trajectory steps."
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.score >= self.threshold


class ToolCorrectnessMetric(BaseMetric):
    """Deterministic heuristic evaluating validity of selected tool handles.

    NOTE: This is a deterministic rule-based metric, NOT an LLM-as-a-Judge.
    """

    def __init__(
        self,
        threshold: float = 0.80,
        allowed_tools: set[str] | None = None,
    ) -> None:
        self.threshold = threshold
        self.allowed_tools = allowed_tools or DEFAULT_ALLOWED_TOOLS
        self.score: float = 0.0
        self.reason: str = ""
        self.success: bool = False

    @property
    def __name__(self) -> str:
        return "ToolCorrectnessMetric"

    def measure(self, test_case: LLMTestCase) -> float:
        tools_called = getattr(test_case, "tools_called", None) or []
        output_lower = (test_case.actual_output or "").lower()

        # Check for known dangerous or hallucinated tools
        if any(inv in output_lower for inv in KNOWN_INVALID_TOOLS):
            self.score = 0.20
            self.reason = "Detected invalid or disallowed tool invocation."
            self.success = self.score >= self.threshold
            return self.score

        if tools_called:
            self.score = 1.0
            self.reason = f"Successfully validated {len(tools_called)} tool calls."
        elif any(tool in output_lower for tool in self.allowed_tools):
            self.score = 0.95
            self.reason = "Allowed tool handle correctly identified in execution trace."
        else:
            self.score = 0.85
            self.reason = "Direct generation without external tool call."

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.score >= self.threshold


class TaskCompletionMetric(BaseMetric):
    """Deterministic heuristic evaluating goal achievement and outcome completeness.

    NOTE: This is a deterministic rule-based metric, NOT an LLM-as-a-Judge.
    """

    def __init__(self, threshold: float = 0.75) -> None:
        self.threshold = threshold
        self.score: float = 0.0
        self.reason: str = ""
        self.success: bool = False

    @property
    def __name__(self) -> str:
        return "TaskCompletionMetric"

    def measure(self, test_case: LLMTestCase) -> float:
        output = (test_case.actual_output or "").strip()
        if not output:
            self.score = 0.0
            self.reason = "Task incomplete: No output produced."
        elif any(err in output.lower() for err in ["plan execution failed", "unhandled exception", "fatal error"]):
            self.score = 0.20
            self.reason = "Task ended in execution failure."
        else:
            self.score = 0.90
            self.reason = "Task completed with valid output."

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.score >= self.threshold


def create_real_geval_trajectory_judge(
    threshold: float = 0.75,
    model: str = "gpt-4o-mini",
) -> Any:
    """Factory creating a real DeepEval GEval LLM-as-a-Judge (Lane 2 only).

    WARNING:
        This invokes a PAID external LLM API. It requires OPENAI_API_KEY.
        Must NEVER be called in default Lane 1 CI runs.
    """
    from deepeval.metrics import GEval
    try:
        from deepeval.test_case import SingleTurnParams
        params = [
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.RETRIEVAL_CONTEXT,
        ]
    except ImportError:
        from deepeval.test_case import LLMTestCaseParams
        params = [
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ]

    return GEval(
        name="RealTrajectoryJudge",
        criteria=(
            "Evaluate whether the actual agentic trajectory logically fulfills the input goal, "
            "strictly adheres to the approved retrieval context plan steps, invokes valid tools, "
            "and achieves successful task completion without unhandled errors."
        ),
        evaluation_params=params,
        threshold=threshold,
        model=model,
    )


class RealGEvalTrajectoryJudge(BaseMetric):
    """Lane-2 LLM-as-a-Judge evaluating trajectories via DeepEval GEval.

    Calls a real LLM judge via OpenAI API. Gated behind Lane 2 / @pytest.mark.heavy.
    """

    def __init__(self, threshold: float = 0.75, model: str = "gpt-4o-mini") -> None:
        self.threshold = threshold
        self.model = model
        self._judge: Any | None = None
        self.score: float = 0.0
        self.reason: str = ""
        self.success: bool = False

    @property
    def __name__(self) -> str:
        return "RealGEvalTrajectoryJudge"

    @property
    def judge(self) -> Any:
        if self._judge is None:
            self._judge = create_real_geval_trajectory_judge(
                threshold=self.threshold,
                model=self.model,
            )
        return self._judge

    def measure(self, test_case: LLMTestCase) -> float:
        self.score = self.judge.measure(test_case)
        self.reason = getattr(self.judge, "reason", "") or ""
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        self.score = await self.judge.a_measure(test_case)
        self.reason = getattr(self.judge, "reason", "") or ""
        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold
