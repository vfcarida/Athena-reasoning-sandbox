"""Integration and Unit Test Suite for Agent-Loop Wiring (ARS-T05).

Tests:
1. End-to-end authorized action execution:
   - Athena query (dry-run and mock)
   - Sandboxed code execution
2. Deny-by-default authorization:
   - Unauthorized tool execution rejected
   - Unregistered tool execution rejected
3. Sandbox isolation enforcement:
   - Default configuration refuses unisolated execution
   - E1 adversarial attempt does not leak environment secrets
4. Failure propagation and early termination:
   - Step failure halts subsequent execution
5. Pydantic validation:
   - Schema enforcement (frozen, extra="forbid")
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.athena.athena_client import AthenaClient
from src.engine.agent_loop import AgentLoop, AgentLoopResult, AgentPlanner
from src.engine.executor import ExecutiveEngineProcess
from src.engine.grpc_boundary import ParallaxToolDispatcher
from src.reasoning.schemas import ActionType, AgentPlan, ReasoningStep
from src.sandbox.e2b_sandbox import E2BSandboxEngine

# ==============================================================================
# 1. Pydantic Schema & Planner Tests
# ==============================================================================


def test_agent_plan_immutability():
    """Test that AgentPlan and ReasoningStep are immutable (frozen)."""
    step = ReasoningStep(
        step_number=1,
        rationale="Test step rationale",
        action_type=ActionType.ATHENA_QUERY,
        tool_name="athena_query",
        tool_args={"sql": "SELECT 1"},
    )
    plan = AgentPlan(
        plan_id="plan-001",
        task_goal="Run a test query",
        steps=[step],
        estimated_complexity=2,
    )

    with pytest.raises(ValidationError):
        # Attempt to mutate frozen model
        plan.task_goal = "Modified goal"  # type: ignore

    with pytest.raises(ValidationError):
        step.rationale = "New rationale"  # type: ignore


def test_agent_plan_forbids_extra_fields():
    """Test that extra fields are rejected during schema validation."""
    with pytest.raises(ValidationError):
        ReasoningStep(
            step_number=1,
            rationale="Test step rationale",
            action_type=ActionType.ATHENA_QUERY,
            tool_name="athena_query",
            tool_args={},
            invalid_field="bad",  # type: ignore
        )

    with pytest.raises(ValidationError):
        AgentPlan(
            plan_id="plan-002",
            task_goal="Run a test query",
            steps=[],  # also violates min_length=1
            extra_property=123,  # type: ignore
        )


def test_agent_planner_creation_and_estimation():
    """Test AgentPlanner creation helpers and complexity estimation."""
    planner = AgentPlanner()

    complexity = planner.estimate_complexity("Run a complex distributed athena query and benchmark", step_count=3)
    assert 1 <= complexity <= 10
    assert complexity >= 6  # boosted by keywords

    athena_step = planner.create_athena_step(
        step_number=1,
        rationale="Query logs for security audit",
        sql="SELECT user_id FROM logs WHERE dt = '2026-01-01' LIMIT 10;",
        required_partition_keys=["dt"],
        dry_run=True,
    )
    assert athena_step.action_type == ActionType.ATHENA_QUERY
    assert athena_step.tool_name == "athena_query"
    assert athena_step.tool_args["sql"].startswith("SELECT")
    assert athena_step.tool_args["dry_run"] is True

    plan = planner.create_plan(
        task_goal="Audit logs query",
        steps=[athena_step],
    )
    assert plan.plan_id is not None
    assert len(plan.steps) == 1
    assert plan.steps[0] == athena_step


# ==============================================================================
# 2. End-to-End Execution: Athena Query (Authorized)
# ==============================================================================


@pytest.mark.asyncio
async def test_agent_loop_athena_dry_run_success():
    """Test that an authorized Athena query runs through the loop and returns a valid observation."""
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
    assert len(result.step_results) == 1
    assert result.step_results[0].success is True
    assert result.final_output["mode"] == "DRY_RUN"
    assert result.final_output["status"] == "SUCCEEDED"
    assert result.error is None


@pytest.mark.asyncio
async def test_agent_loop_athena_mock_execution():
    """Test that an authorized Athena query with mock boto3 executes and returns query results."""
    mock_boto = MagicMock()
    mock_boto.start_query_execution.return_value = {"QueryExecutionId": "q-12345"}
    mock_boto.get_query_execution.return_value = {
        "QueryExecution": {
            "QueryExecutionId": "q-12345",
            "Status": {"State": "SUCCEEDED"},
            "Statistics": {"DataScannedInBytes": 1048576, "EngineExecutionTimeInMillis": 120},
        }
    }

    athena_client = AthenaClient(boto_client=mock_boto, max_scan_bytes=10485760)
    executor = ExecutiveEngineProcess(athena_client=athena_client)
    dispatcher = ParallaxToolDispatcher(executor=executor)
    loop = AgentLoop(dispatcher=dispatcher)

    planner = AgentPlanner()
    step = planner.create_athena_step(
        step_number=1,
        rationale="Execute mocked query and wait for completion",
        sql="SELECT event_name FROM app_events WHERE dt = '2026-01-15' LIMIT 50;",
        required_partition_keys=["dt"],
        wait=True,
    )
    plan = planner.create_plan(task_goal="Run Athena query", steps=[step])

    result = await loop.run_plan(plan)

    assert result.success is True
    assert result.final_output["status"] == "SUCCEEDED"
    assert result.final_output["data_scanned_in_bytes"] == 1048576
    assert result.final_output["query_execution_id"] == "q-12345"


# ==============================================================================
# 3. End-to-End Execution: Sandboxed Code Execution (Authorized)
# ==============================================================================


@pytest.mark.asyncio
async def test_agent_loop_sandbox_execution_success():
    """Test that an authorized sandboxed code execution runs and captures stdout."""
    sandbox = E2BSandboxEngine(allow_unsafe_local=True)
    executor = ExecutiveEngineProcess(sandbox_engine=sandbox)
    dispatcher = ParallaxToolDispatcher(executor=executor)
    loop = AgentLoop(dispatcher=dispatcher)

    planner = AgentPlanner()
    step = planner.create_sandbox_step(
        step_number=1,
        rationale="Compute factorial in sandbox",
        code="import math\nprint(math.factorial(6))",
        language="python",
    )
    plan = planner.create_plan(task_goal="Factorial calculation", steps=[step])

    result = await loop.run_plan(plan)

    assert result.success is True
    assert result.error is None
    assert len(result.step_results) == 1
    assert result.step_results[0].success is True
    assert "720" in result.final_output["stdout"]
    assert result.final_output["exit_code"] == 0


# ==============================================================================
# 4. Deny-by-Default and Authorization Enforcement
# ==============================================================================


@pytest.mark.asyncio
async def test_agent_loop_denies_unauthorized_tool():
    """Test that a tool not in the authorized_tools allowlist is rejected."""
    # Allow only "athena_query", not "sandbox_execute"
    executor = ExecutiveEngineProcess(authorized_tools={"athena_query"})
    dispatcher = ParallaxToolDispatcher(executor=executor)
    loop = AgentLoop(dispatcher=dispatcher)

    planner = AgentPlanner()
    step = planner.create_sandbox_step(
        step_number=1,
        rationale="Try to run unpermitted sandbox tool",
        code="print('should not run')",
    )
    plan = planner.create_plan(task_goal="Unauthorized execution", steps=[step])

    result = await loop.run_plan(plan)

    assert result.success is False
    assert "Unauthorized tool execution" in result.error
    assert "sandbox_execute" in result.error
    assert result.step_results[0].success is False


@pytest.mark.asyncio
async def test_agent_loop_denies_unregistered_tool():
    """Test that an unregistered tool is rejected even if in allowlist."""
    executor = ExecutiveEngineProcess(authorized_tools={"custom_tool"})
    dispatcher = ParallaxToolDispatcher(executor=executor)
    loop = AgentLoop(dispatcher=dispatcher)

    step = ReasoningStep(
        step_number=1,
        rationale="Invoke non-existent tool",
        action_type=ActionType.RETRIEVAL_SEARCH,
        tool_name="custom_tool",
        tool_args={},
    )
    plan = AgentPlan(plan_id="plan-unregistered", task_goal="Test unregistered", steps=[step])

    result = await loop.run_plan(plan)

    assert result.success is False
    assert "Unregistered executive tool" in result.error


# ==============================================================================
# 5. Isolate-or-Refuse and E1 Security
# ==============================================================================


@pytest.mark.asyncio
async def test_agent_loop_sandbox_refuses_when_unisolated():
    """Test that sandbox_execute refuses to run without verified isolation (ARS-T04)."""
    # Default sandbox engine without api_key and allow_unsafe_local=False
    sandbox = E2BSandboxEngine(api_key="", allow_unsafe_local=False)

    # Check if host has container runtime; if not, must refuse
    if not sandbox.is_container_available() and not sandbox.is_gvisor_available():
        executor = ExecutiveEngineProcess(sandbox_engine=sandbox)
        dispatcher = ParallaxToolDispatcher(executor=executor)
        loop = AgentLoop(dispatcher=dispatcher)

        planner = AgentPlanner()
        step = planner.create_sandbox_step(
            step_number=1,
            rationale="Attempt unisolated execution",
            code="print('unsafe')",
        )
        plan = planner.create_plan(task_goal="Unsafe test", steps=[step])

        result = await loop.run_plan(plan)

        assert result.success is False
        assert "SandboxUnavailableError" in result.error or "Refusing" in result.error


@pytest.mark.asyncio
async def test_agent_loop_e1_environment_isolation():
    """Test that E1 attack (attempting to read host credentials) fails due to env sanitization."""
    os.environ["AWS_SECRET_ACCESS_KEY"] = "super-secret-key-12345"
    os.environ["E2B_API_KEY"] = "super-secret-e2b-token"

    try:
        sandbox = E2BSandboxEngine(allow_unsafe_local=True)
        executor = ExecutiveEngineProcess(sandbox_engine=sandbox)
        dispatcher = ParallaxToolDispatcher(executor=executor)
        loop = AgentLoop(dispatcher=dispatcher)

        # Code attempts to print secret environment variables
        code = (
            "import os\n"
            "print('AWS_KEY:', os.environ.get('AWS_SECRET_ACCESS_KEY', 'NOT_FOUND'))\n"
            "print('E2B_KEY:', os.environ.get('E2B_API_KEY', 'NOT_FOUND'))\n"
        )

        planner = AgentPlanner()
        step = planner.create_sandbox_step(
            step_number=1,
            rationale="Verify credential isolation",
            code=code,
        )
        plan = planner.create_plan(task_goal="Security test", steps=[step])

        result = await loop.run_plan(plan)

        assert result.success is True
        assert "AWS_KEY: NOT_FOUND" in result.final_output["stdout"]
        assert "E2B_KEY: NOT_FOUND" in result.final_output["stdout"]
        assert "super-secret-key-12345" not in result.final_output["stdout"]
        assert "super-secret-e2b-token" not in result.final_output["stdout"]
    finally:
        os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
        os.environ.pop("E2B_API_KEY", None)


# ==============================================================================
# 6. Error Propagation and Multi-Step Execution
# ==============================================================================


@pytest.mark.asyncio
async def test_agent_loop_fails_fast_on_error():
    """Test that a failure in an early step stops further execution and returns error."""
    athena_client = AthenaClient()
    executor = ExecutiveEngineProcess(athena_client=athena_client)
    dispatcher = ParallaxToolDispatcher(executor=executor)
    loop = AgentLoop(dispatcher=dispatcher)

    # Step 1: Illegal unpartitioned query
    step1 = ReasoningStep(
        step_number=1,
        rationale="Run invalid query without partition filter",
        action_type=ActionType.ATHENA_QUERY,
        tool_name="athena_query",
        tool_args={
            "sql": "SELECT * FROM large_table WHERE status = 'ACTIVE' LIMIT 10",
            "required_partition_keys": ["dt"],
            "dry_run": True,
        },
    )
    # Step 2: Ping
    step2 = ReasoningStep(
        step_number=2,
        rationale="Follow-up ping that should not be reached",
        action_type=ActionType.FINAL_ANSWER,
        tool_name="ping",
        tool_args={"msg": "ping"},
    )

    plan = AgentPlan(
        plan_id="plan-fail-fast",
        task_goal="Fail-fast demonstration",
        steps=[step1, step2],
    )

    result = await loop.run_plan(plan)

    assert result.success is False
    assert len(result.step_results) == 1
    assert "UnpartitionedQueryError" in result.error or "missing required partition filter" in result.error
    assert result.final_output is None


@pytest.mark.asyncio
async def test_agent_loop_multi_step_success():
    """Test a valid multi-step plan (Athena query followed by sandbox processing)."""
    athena_client = AthenaClient()
    sandbox = E2BSandboxEngine(allow_unsafe_local=True)
    executor = ExecutiveEngineProcess(
        athena_client=athena_client,
        sandbox_engine=sandbox,
    )
    dispatcher = ParallaxToolDispatcher(executor=executor)
    loop = AgentLoop(dispatcher=dispatcher)

    planner = AgentPlanner()
    step1 = planner.create_athena_step(
        step_number=1,
        rationale="Query table partition",
        sql="SELECT col1 FROM table1 WHERE dt = '2026-02-01' LIMIT 10",
        required_partition_keys=["dt"],
        dry_run=True,
    )
    step2 = planner.create_sandbox_step(
        step_number=2,
        rationale="Process query results",
        code="print('Processed 10 records')",
    )

    plan = planner.create_plan(
        task_goal="Multi-step workflow",
        steps=[step1, step2],
    )

    result = await loop.run_plan(plan)

    assert result.success is True
    assert len(result.step_results) == 2
    assert result.step_results[0].success is True
    assert result.step_results[1].success is True
    assert "Processed 10 records" in result.final_output["stdout"]
