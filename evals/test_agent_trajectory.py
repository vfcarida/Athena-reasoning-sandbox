"""Automated DeepEval Trajectory Evaluation Test Suite.

Evaluates agentic reasoning trajectories against quality metrics:
- PlanQualityMetric (Completeness & Logic)
- PlanAdherenceMetric (Strategy adherence)
- ToolCorrectnessMetric (Tool handle validity)
- TaskCompletionMetric (Goal achievement)

Run with:
    deepeval test run evals/test_agent_trajectory.py
"""

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from evals.metrics.trajectory_metrics import (
    PlanQualityMetric,
    PlanAdherenceMetric,
    ToolCorrectnessMetric,
    TaskCompletionMetric,
)


@pytest.mark.asyncio
async def test_agent_athena_query_trajectory():
    """Evaluate an agentic trajectory for querying AWS Athena."""
    test_case = LLMTestCase(
        input="Query user events log table in Athena for date 2026-08-12 and summarize total active users.",
        actual_output=(
            "Step 1: Check partition strategy for user_events table.\n"
            "Step 2: Execute cost-guarded query: SELECT user_id, status FROM user_events WHERE dt = '2026-08-12' LIMIT 1000;\n"
            "Step 3: Aggregate total unique user_id count and output summary."
        ),
        retrieval_context=[
            "Step 1: Validate partition filters.",
            "Step 2: Query user_events where dt = '2026-08-12'.",
            "Step 3: Return summary count.",
        ],
    )

    plan_quality = PlanQualityMetric(threshold=0.75)
    plan_adherence = PlanAdherenceMetric(threshold=0.75)
    tool_correctness = ToolCorrectnessMetric(threshold=0.75)
    task_completion = TaskCompletionMetric(threshold=0.75)

    assert_test(
        test_case=test_case,
        metrics=[plan_quality, plan_adherence, tool_correctness, task_completion],
    )


@pytest.mark.asyncio
async def test_agent_swi_reasoning_trajectory():
    """Evaluate SwiReasoning entropy-guided generation trajectory."""
    test_case = LLMTestCase(
        input="Explain how spherical linear interpolation (SLERP) preserves weight norm geometry during LLM merging.",
        actual_output=(
            "<think>SLERP interpolates along the unit hypersphere surface using angular distance theta.</think>"
            "SLERP (Spherical Linear Interpolation) performs geometric interpolation along the high-dimensional "
            "hypersphere of normalized parameter matrices, maintaining angular velocity and preventing magnitude decay."
        ),
        retrieval_context=[
            "Explain SLERP math.",
            "Describe angular interpolation on hypersphere.",
        ],
    )

    plan_quality = PlanQualityMetric(threshold=0.70)
    task_completion = TaskCompletionMetric(threshold=0.70)

    assert_test(
        test_case=test_case,
        metrics=[plan_quality, task_completion],
    )
