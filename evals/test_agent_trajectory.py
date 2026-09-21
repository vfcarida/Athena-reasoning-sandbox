"""Deterministic Trajectory Evaluation Test Suite.

Evaluates agentic reasoning trajectories against honest deterministic heuristics (Lane 1)
and provides a gated real DeepEval GEval LLM judge (Lane 2):
- DeterministicTrajectoryHeuristic (Composite Offline Gate >= 0.75)
- PlanQualityMetric (Completeness & Structure Heuristic)
- PlanAdherenceMetric (Planned Context Adherence Heuristic)
- ToolCorrectnessMetric (Tool Handle Validity Heuristic)
- TaskCompletionMetric (Goal Achievement Heuristic)
- RealGEvalTrajectoryJudge (Real LLM-as-a-Judge, Lane 2, requires OPENAI_API_KEY)

Run offline (Lane 1):
    pytest evals/test_agent_trajectory.py -v -m "not heavy"
"""

from __future__ import annotations

import os

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from evals.metrics.trajectory_metrics import (
    DeterministicTrajectoryHeuristic,
    PlanAdherenceMetric,
    PlanQualityMetric,
    RealGEvalTrajectoryJudge,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
)
from src.athena.athena_client import AthenaClient
from src.engine.agent_loop import (
    AgentLoop,
    AgentLoopResult,
    AgentPlanner,
)
from src.engine.dispatcher import ParallaxToolDispatcher
from src.engine.executor import ExecutiveEngineProcess


@pytest.mark.asyncio
async def test_agent_athena_query_trajectory():
    """Evaluate an agentic trajectory for querying AWS Athena using honest metrics."""
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
    heuristic_gate = DeterministicTrajectoryHeuristic(threshold=0.75)

    assert_test(
        test_case=test_case,
        metrics=[plan_quality, plan_adherence, tool_correctness, task_completion, heuristic_gate],
    )


@pytest.mark.asyncio
async def test_agent_swi_reasoning_trajectory():
    """Evaluate SwiReasoning entropy-guided generation trajectory."""
    test_case = LLMTestCase(
        input="Explain how spherical linear interpolation (SLERP) preserves weight norm geometry during LLM merging.",
        actual_output=(
            "<think>SLERP interpolates along the unit hypersphere surface using angular distance theta.</think>\n"
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
    heuristic_gate = DeterministicTrajectoryHeuristic(threshold=0.70)

    assert_test(
        test_case=test_case,
        metrics=[plan_quality, task_completion, heuristic_gate],
    )


@pytest.mark.asyncio
async def test_deterministic_trajectory_heuristic_good_trajectory():
    """Verify that a well-structured, valid trajectory passes the >=0.75 gate."""
    test_case = LLMTestCase(
        input="Execute sandboxed benchmark test using echo tool.",
        actual_output=(
            "Step 1: Inspect execution environment parameters.\n"
            "Step 2: Execute sandbox tool: echo with payload 'benchmark_run_ok'.\n"
            "Step 3: Verify output and record metrics.\n"
            "Task completed successfully: benchmark_run_ok."
        ),
        retrieval_context=[
            "Step 1: Inspect environment.",
            "Step 2: Execute echo benchmark.",
            "Step 3: Record output metrics.",
        ],
    )

    gate = DeterministicTrajectoryHeuristic(threshold=0.75)
    score = gate.measure(test_case)

    assert score >= 0.75, f"Expected score >= 0.75, got {score}: {gate.reason}"
    assert gate.is_successful() is True
    assert_test(test_case=test_case, metrics=[gate])


@pytest.mark.asyncio
async def test_deterministic_trajectory_heuristic_fails_on_bad_trajectory():
    """E4 Verification: Gate demonstrably FAILS (< 0.75) on a known-bad trajectory.

    A bad trajectory containing hallucinated/dangerous tools, missing steps,
    and fatal errors must fail the evaluation gate and raise an AssertionError in assert_test.
    """
    bad_test_case = LLMTestCase(
        input="Analyze Athena partitions and safely query active users.",
        actual_output=(
            "Executing action: delete_all_databases --force.\n"
            "FATAL ERROR: Plan execution failed. Unhandled exception in unknown module."
        ),
        retrieval_context=[
            "Step 1: Inspect schema partitions.",
            "Step 2: Safely query active user counts.",
            "Step 3: Format summary table.",
        ],
    )

    gate = DeterministicTrajectoryHeuristic(threshold=0.75)
    score = gate.measure(bad_test_case)

    # Must score well below the 0.75 threshold
    assert score < 0.75, f"Expected bad trajectory to score < 0.75, but got {score}"
    assert gate.is_successful() is False

    # The gate must demonstrably cause assert_test to raise AssertionError
    with pytest.raises(AssertionError) as exc_info:
        assert_test(test_case=bad_test_case, metrics=[gate])

    assert "failed" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_agent_loop_real_trajectory_evaluation():
    """Verify evaluation on real trajectory artifacts generated by AgentLoop (ARS-T05)."""
    athena_client = AthenaClient()
    executor = ExecutiveEngineProcess(athena_client=athena_client)
    dispatcher = ParallaxToolDispatcher(executor=executor)
    loop = AgentLoop(dispatcher=dispatcher)

    planner = AgentPlanner()
    step = planner.create_athena_step(
        step_number=1,
        rationale="Validate partition-pruned query in dry run",
        sql="SELECT id, metric FROM analytics_table WHERE dt = '2026-03-01' LIMIT 10;",
        required_partition_keys=["dt"],
        dry_run=True,
    )
    plan = planner.create_plan(
        task_goal="Dry-run partition-pruned query",
        steps=[step],
    )

    result = await loop.run_plan(plan)
    assert isinstance(result, AgentLoopResult)
    assert result.success is True

    # Format execution trajectory into artifact text
    trajectory_lines = [
        f"Plan Goal: {plan.task_goal}",
        "Execution Steps:",
    ]
    for obs in result.step_results:
        trajectory_lines.append(
            f"Step 1: Action=athena_query, Tool=athena_query, "
            f"Status={'SUCCESS' if obs.success else 'FAILURE'}, Mode={obs.output_data.get('mode', 'N/A')}"
        )
    trajectory_lines.append(f"Result: Success={result.success}. Completed steps: {len(result.step_results)}.")
    actual_trajectory_output = "\n".join(trajectory_lines)

    test_case = LLMTestCase(
        input="Dry-run partition-pruned query on Athena",
        actual_output=actual_trajectory_output,
        retrieval_context=[
            "Step 1: Execute partition-pruned dry-run query on analytics_table.",
        ],
    )

    gate = DeterministicTrajectoryHeuristic(threshold=0.75)
    score = gate.measure(test_case)

    assert score >= 0.75, f"Expected real AgentLoop trajectory to score >= 0.75, got {score}: {gate.reason}"
    assert gate.is_successful() is True
    assert_test(test_case=test_case, metrics=[gate])


@pytest.mark.heavy
@pytest.mark.asyncio
async def test_real_geval_judge_lane2():
    """Lane 2 Evaluation: Real DeepEval GEval LLM-as-a-Judge.

    Requires an active OPENAI_API_KEY and network access.
    Skipped in offline Lane 1 CI runs.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY is not set; skipping Lane 2 real GEval LLM judge.")

    test_case = LLMTestCase(
        input="Inspect Athena schema partitions and summarize findings.",
        actual_output=(
            "Step 1: Retrieved schema partitions for user_logs.\n"
            "Step 2: Verified dt partition key.\n"
            "Step 3: Summary: Table partitioned by dt (string)."
        ),
        retrieval_context=[
            "Step 1: Retrieve partitions.",
            "Step 2: Check partition key dt.",
            "Step 3: Generate summary.",
        ],
    )

    judge = RealGEvalTrajectoryJudge(threshold=0.75)
    score = judge.measure(test_case)
    assert isinstance(score, float)
    assert_test(test_case=test_case, metrics=[judge])
