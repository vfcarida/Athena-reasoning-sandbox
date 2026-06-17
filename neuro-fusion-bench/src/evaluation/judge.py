"""LLM-as-a-Judge — Semantic evaluation of agent plans via language model reasoning.

This module implements a judge system that uses a language model to evaluate
the quality and adherence of AI agent execution plans. Unlike the deterministic
metrics in :mod:`agent_bench`, the LLM Judge provides semantic understanding
of plan quality — assessing coherence, feasibility, and goal alignment.

The judge operates in two modes:
    - **Online Mode**: Uses a Hugging Face ``pipeline`` for local LLM inference.
    - **Offline/Mock Mode**: Returns simulated scores for testing and demos.

Each evaluation returns a structured result with a numeric score [0.0, 1.0]
and a textual reasoning chain explaining the assessment.

References:
    - Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────────────────────────────────────

PLAN_QUALITY_PROMPT = """You are an impartial evaluation judge assessing the quality of an AI agent's plan.

## Objective
{objective}

## Agent's Plan
{plan_json}

## Evaluation Criteria
Assess the plan on the following dimensions and provide a single composite score:

1. **Coherence** (0.0–1.0): Are the steps logically ordered and internally consistent?
2. **Feasibility** (0.0–1.0): Can each step be realistically executed with available tools?
3. **Completeness** (0.0–1.0): Does the plan cover all necessary steps to achieve the objective?
4. **Efficiency** (0.0–1.0): Is the plan concise without unnecessary steps?

## Response Format
Respond with ONLY a valid JSON object (no markdown, no explanation outside JSON):
{{"score": <float 0.0 to 1.0>, "reasoning": "<brief explanation>", "coherence": <float>, "feasibility": <float>, "completeness": <float>, "efficiency": <float>}}
"""

PLAN_ADHERENCE_PROMPT = """You are an impartial evaluation judge assessing how well an AI agent followed its declared plan during execution.

## Agent's Declared Plan
{plan_json}

## Actual Execution Trace
{trace_json}

## Evaluation Criteria
1. **Step Coverage**: Were all planned steps executed?
2. **Order Fidelity**: Were steps executed in the planned order?
3. **Deviation Severity**: How significant were any deviations from the plan?
4. **Adaptive Quality**: If deviations occurred, were they reasonable adaptations?

## Response Format
Respond with ONLY a valid JSON object (no markdown, no explanation outside JSON):
{{"score": <float 0.0 to 1.0>, "reasoning": "<brief explanation>", "coverage": <float>, "order_fidelity": <float>, "deviation_severity": <float>}}
"""


@dataclass
class JudgeResult:
    """Container for LLM Judge evaluation output.

    Attributes:
        score: Overall quality/adherence score in [0.0, 1.0].
        reasoning: Textual explanation of the score.
        sub_scores: Dictionary of dimension-specific scores (e.g., coherence,
            feasibility, etc.).
        raw_response: The raw text response from the LLM (for debugging).
    """
    score: float
    reasoning: str
    sub_scores: dict[str, float]
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary for reporting."""
        return {
            "score": round(self.score, 4),
            "reasoning": self.reasoning,
            "sub_scores": {k: round(v, 4) for k, v in self.sub_scores.items()},
        }


class LLMJudge:
    """LLM-based semantic evaluation judge for agent plan quality and adherence.

    Uses a language model to provide nuanced, context-aware scoring of agent
    behaviors that cannot be captured by purely deterministic metrics.

    The judge supports two operational modes:
    - **Online**: Real LLM inference via Hugging Face ``transformers``.
    - **Offline**: Deterministic mock scoring for testing and demonstrations.

    Example:
        >>> judge = LLMJudge(mode="offline")
        >>> result = judge.evaluate_plan_quality(
        ...     objective="Book a flight to Tokyo",
        ...     plan_json='["search_flights", "compare_prices", "book_ticket"]',
        ... )
        >>> print(f"Quality: {result.score:.2f} — {result.reasoning}")
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        temperature: float = 0.1,
        max_tokens: int = 512,
        mode: str = "offline",
        device: Optional[str] = None,
    ) -> None:
        """Initialize the LLM Judge.

        Args:
            model_name: Hugging Face model identifier for the judge LLM.
                Only used when ``mode="online"``.
            temperature: Sampling temperature for LLM generation. Lower values
                produce more deterministic evaluations.
            max_tokens: Maximum tokens for the judge's response.
            mode: Operating mode. ``"online"`` for real LLM inference,
                ``"offline"`` for mock/simulated scoring.
            device: Target device for model inference. Auto-detected if None.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.mode = mode
        self.device = device
        self._pipeline = None

        if mode == "online":
            self._initialize_pipeline()

        logger.info(
            "LLMJudge initialized: mode=%s, model=%s, temperature=%.2f",
            mode, model_name, temperature,
        )

    def _initialize_pipeline(self) -> None:
        """Initialize the Hugging Face text-generation pipeline.

        Lazily loads the model and tokenizer. Catches import and loading errors
        gracefully, falling back to offline mode if the model cannot be loaded.
        """
        try:
            from transformers import pipeline as hf_pipeline  # noqa: F811

            device_arg = self.device
            if device_arg is None:
                import torch
                device_arg = "cuda" if torch.cuda.is_available() else "cpu"

            self._pipeline = hf_pipeline(
                "text-generation",
                model=self.model_name,
                device=device_arg,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0.0,
                return_full_text=False,
            )
            logger.info("LLM pipeline loaded: %s on %s", self.model_name, device_arg)

        except Exception as e:
            logger.warning(
                "Failed to load LLM pipeline (%s). Falling back to offline mode. "
                "Error: %s", self.model_name, e,
            )
            self.mode = "offline"
            self._pipeline = None

    def _query_llm(self, prompt: str) -> str:
        """Send a prompt to the LLM and return the raw text response.

        Args:
            prompt: The full evaluation prompt.

        Returns:
            Raw text response from the LLM.

        Raises:
            RuntimeError: If the pipeline is not initialized.
        """
        if self._pipeline is None:
            raise RuntimeError("LLM pipeline is not initialized.")

        outputs = self._pipeline(prompt)
        if outputs and isinstance(outputs, list) and len(outputs) > 0:
            return outputs[0].get("generated_text", "")
        return ""

    def _parse_judge_response(self, raw_response: str) -> dict[str, Any]:
        """Parse the LLM's JSON response, with robust error handling.

        Attempts to extract a JSON object from the response text. Handles common
        issues like markdown code fences and extra text around the JSON.

        Args:
            raw_response: Raw text from the LLM.

        Returns:
            Parsed dictionary. Returns a default structure on parse failure.
        """
        # Strip markdown code fences if present
        cleaned = raw_response.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        # Try direct JSON parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the text
        json_match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: return default structure
        logger.warning("Failed to parse judge response as JSON: %s", raw_response[:200])
        return {
            "score": 0.5,
            "reasoning": "Unable to parse judge response. Defaulting to neutral score.",
        }

    def _mock_plan_quality(
        self,
        objective: str,
        plan_json: str,
    ) -> JudgeResult:
        """Generate a deterministic mock evaluation for plan quality.

        Produces scores based on heuristic analysis of plan structure:
        - Plan length relative to objective complexity
        - Presence of diverse step types
        - Logical ordering indicators

        Args:
            objective: The task objective.
            plan_json: JSON representation of the agent's plan.

        Returns:
            :class:`JudgeResult` with heuristic-based scores.
        """
        try:
            plan = json.loads(plan_json) if isinstance(plan_json, str) else plan_json
        except json.JSONDecodeError:
            plan = []

        if not isinstance(plan, list):
            plan = [str(plan)]

        num_steps = len(plan)
        unique_steps = len(set(str(s) for s in plan))

        # Heuristic scoring
        completeness = min(num_steps / max(3, 1), 1.0)
        efficiency = unique_steps / max(num_steps, 1)
        coherence = 0.8 if num_steps > 1 else 0.5
        feasibility = 0.85  # Assume reasonable feasibility for well-formed plans

        composite = (coherence + feasibility + completeness + efficiency) / 4.0

        reasoning = (
            f"Plan has {num_steps} steps with {unique_steps} unique actions. "
            f"Structure appears {'well-organized' if composite > 0.7 else 'needs improvement'}. "
            f"Coherence={coherence:.2f}, Feasibility={feasibility:.2f}, "
            f"Completeness={completeness:.2f}, Efficiency={efficiency:.2f}."
        )

        return JudgeResult(
            score=round(composite, 4),
            reasoning=reasoning,
            sub_scores={
                "coherence": coherence,
                "feasibility": feasibility,
                "completeness": completeness,
                "efficiency": efficiency,
            },
            raw_response="[MOCK MODE]",
        )

    def _mock_plan_adherence(
        self,
        plan_json: str,
        trace_json: str,
    ) -> JudgeResult:
        """Generate a deterministic mock evaluation for plan adherence.

        Uses simple sequence comparison heuristics to estimate adherence.

        Args:
            plan_json: JSON representation of the declared plan.
            trace_json: JSON representation of the execution trace.

        Returns:
            :class:`JudgeResult` with heuristic-based adherence scores.
        """
        try:
            plan = json.loads(plan_json) if isinstance(plan_json, str) else plan_json
        except json.JSONDecodeError:
            plan = []

        try:
            trace = json.loads(trace_json) if isinstance(trace_json, str) else trace_json
        except json.JSONDecodeError:
            trace = []

        if not isinstance(plan, list):
            plan = [str(plan)]
        if not isinstance(trace, list):
            trace = [str(trace)]

        plan_set = set(str(s) for s in plan)
        trace_set = set(str(s) for s in trace)

        # Coverage: how many plan steps appear in trace
        if plan_set:
            coverage = len(plan_set & trace_set) / len(plan_set)
        else:
            coverage = 1.0

        # Order fidelity: simple positional comparison
        order_score = 0.0
        if plan and trace:
            matches = 0
            trace_idx = 0
            for step in plan:
                step_str = str(step)
                while trace_idx < len(trace):
                    if str(trace[trace_idx]) == step_str:
                        matches += 1
                        trace_idx += 1
                        break
                    trace_idx += 1
            order_score = matches / len(plan) if plan else 1.0

        # Deviation severity (lower is better, inverted for scoring)
        extra_steps = len(trace_set - plan_set)
        deviation = 1.0 - min(extra_steps / max(len(plan), 1), 1.0)

        composite = (coverage * 0.4 + order_score * 0.4 + deviation * 0.2)

        reasoning = (
            f"Coverage: {coverage:.2f} of plan steps executed. "
            f"Order fidelity: {order_score:.2f}. "
            f"Deviation severity: {1.0 - deviation:.2f} "
            f"({extra_steps} extra step(s) not in plan)."
        )

        return JudgeResult(
            score=round(composite, 4),
            reasoning=reasoning,
            sub_scores={
                "coverage": round(coverage, 4),
                "order_fidelity": round(order_score, 4),
                "deviation_severity": round(deviation, 4),
            },
            raw_response="[MOCK MODE]",
        )

    def evaluate_plan_quality(
        self,
        objective: str,
        plan_json: str,
    ) -> JudgeResult:
        """Evaluate the quality of an agent's plan against a stated objective.

        Assesses coherence, feasibility, completeness, and efficiency of the
        plan through LLM reasoning (online mode) or heuristic analysis (offline).

        Args:
            objective: Natural language description of the task goal.
            plan_json: JSON-serialized representation of the agent's plan
                (typically a list of step descriptions or action dictionaries).

        Returns:
            :class:`JudgeResult` with quality score and dimensional breakdown.
        """
        logger.info("Evaluating plan quality (mode=%s)...", self.mode)

        if self.mode == "offline":
            return self._mock_plan_quality(objective, plan_json)

        # Online mode: query the LLM
        prompt = PLAN_QUALITY_PROMPT.format(
            objective=objective,
            plan_json=plan_json,
        )

        raw_response = self._query_llm(prompt)
        parsed = self._parse_judge_response(raw_response)

        score = float(parsed.get("score", 0.5))
        score = max(0.0, min(1.0, score))

        sub_scores = {
            "coherence": float(parsed.get("coherence", score)),
            "feasibility": float(parsed.get("feasibility", score)),
            "completeness": float(parsed.get("completeness", score)),
            "efficiency": float(parsed.get("efficiency", score)),
        }

        return JudgeResult(
            score=score,
            reasoning=parsed.get("reasoning", "No reasoning provided."),
            sub_scores=sub_scores,
            raw_response=raw_response,
        )

    def evaluate_plan_adherence(
        self,
        plan_json: str,
        trace_json: str,
    ) -> JudgeResult:
        """Evaluate how closely the agent followed its declared plan.

        Compares the intended plan against the actual execution trace to assess
        step coverage, execution order, and deviation severity.

        Args:
            plan_json: JSON-serialized representation of the agent's declared plan.
            trace_json: JSON-serialized representation of the actual execution trace.

        Returns:
            :class:`JudgeResult` with adherence score and dimensional breakdown.
        """
        logger.info("Evaluating plan adherence (mode=%s)...", self.mode)

        if self.mode == "offline":
            return self._mock_plan_adherence(plan_json, trace_json)

        # Online mode: query the LLM
        prompt = PLAN_ADHERENCE_PROMPT.format(
            plan_json=plan_json,
            trace_json=trace_json,
        )

        raw_response = self._query_llm(prompt)
        parsed = self._parse_judge_response(raw_response)

        score = float(parsed.get("score", 0.5))
        score = max(0.0, min(1.0, score))

        sub_scores = {
            "coverage": float(parsed.get("coverage", score)),
            "order_fidelity": float(parsed.get("order_fidelity", score)),
            "deviation_severity": float(parsed.get("deviation_severity", score)),
        }

        return JudgeResult(
            score=score,
            reasoning=parsed.get("reasoning", "No reasoning provided."),
            sub_scores=sub_scores,
            raw_response=raw_response,
        )
