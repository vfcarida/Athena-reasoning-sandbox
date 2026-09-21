"""Unit test suite for Phase 1: Cognitive-Executive Separation & Plan-then-Execute.

Tests Pydantic output validation, asynchronous execution, and Parallax API boundary isolation.
"""


import pytest

from src.engine.dispatcher import ParallaxToolDispatcher
from src.engine.executor import ExecutiveEngineProcess
from src.reasoning.schemas import (
    ActionType,
    AgentPlan,
    ObservationPayload,
    ReasoningStep,
    ToolCallPayload,
)


@pytest.mark.asyncio
async def test_pydantic_schema_validation():
    """Verify type-safe Pydantic schema validation for agent plans and tool calls."""
    step = ReasoningStep(
        step_number=1,
        rationale="Retrieve partition info from Athena.",
        action_type=ActionType.ATHENA_QUERY,
        tool_name="athena_query",
        tool_args={"query": "SELECT count(*) FROM logs WHERE dt = '2026-08-12';"},
    )
    assert step.step_number == 1
    assert step.action_type == ActionType.ATHENA_QUERY

    plan = AgentPlan(
        plan_id="plan-12345",
        task_goal="Query AWS Athena data",
        thinking_process="Analyzing partition strategy...",
        steps=[step],
        estimated_complexity=3,
    )
    assert plan.plan_id == "plan-12345"
    assert len(plan.steps) == 1


@pytest.mark.asyncio
async def test_executive_engine_process_ping():
    """Test ExecutiveEngineProcess execution of registered tools."""
    executor = ExecutiveEngineProcess()
    payload = ToolCallPayload(
        call_id="call-001",
        tool_name="ping",
        arguments={"test": "val"},
        timeout_seconds=5.0,
    )
    obs: ObservationPayload = await executor.execute_tool_call(payload)
    assert obs.success is True
    assert obs.output_data["status"] == "pong"
    assert obs.call_id == "call-001"


@pytest.mark.asyncio
async def test_executive_engine_unregistered_tool():
    """Test handling of unregistered tools in Executive Engine."""
    executor = ExecutiveEngineProcess()
    payload = ToolCallPayload(
        call_id="call-002",
        tool_name="unregistered_tool_name",
        arguments={},
        timeout_seconds=5.0,
    )
    obs = await executor.execute_tool_call(payload)
    assert obs.success is False
    assert "Unregistered" in obs.error_message


@pytest.mark.asyncio
async def test_parallax_tool_dispatcher():
    """Test ParallaxToolDispatcher boundary isolation."""
    executor = ExecutiveEngineProcess()
    dispatcher = ParallaxToolDispatcher(executor=executor)

    payload = ToolCallPayload(
        call_id="call-003",
        tool_name="echo",
        arguments={"message": "hello boundary"},
        timeout_seconds=5.0,
    )
    obs = await dispatcher.dispatch_tool_call(payload)
    assert obs.success is True
    assert obs.output_data["echo"] == "hello boundary"


def test_grpc_boundary_deprecation_warning():
    """Verify that importing from src.engine.grpc_boundary emits a DeprecationWarning."""
    import importlib
    import sys
    import warnings

    if "src.engine.grpc_boundary" in sys.modules:
        del sys.modules["src.engine.grpc_boundary"]

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        mod = importlib.import_module("src.engine.grpc_boundary")
        assert hasattr(mod, "ParallaxToolDispatcher")
        assert any(issubclass(w.category, DeprecationWarning) for w in recorded)


@pytest.mark.heavy
@pytest.mark.asyncio
async def test_swi_reasoning_async_simulation():
    """Test async SwiReasoning simulator and AgentPlan extraction."""
    from src.reasoning.swi_reasoning import SwiReasoningSimulator
    sim = SwiReasoningSimulator(vocab_size=1000, entropy_threshold=2.0, seed=42)
    res = await sim.simulate_async("Explain quantum computing", num_steps=15)
    assert res.state.total_tokens == 15
    plan = res.to_agent_plan()
    assert plan.task_goal == "Explain quantum computing"
    assert len(plan.steps) == 1
