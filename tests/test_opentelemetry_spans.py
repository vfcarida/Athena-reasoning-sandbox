"""Unit test suite for ENG-03: OpenTelemetry GenAI Semantic Convention Spans.

Verifies that:
1. Standard GenAI semantic convention attribute constants are defined and properly formatted.
2. Attribute normalization handles primitives, complex nested objects, and JSON serialization safely.
3. MockSpan and MockSpanExporter provide a clean fallback when OpenTelemetry SDK is not in use.
4. ExecutiveEngineProcess emits structured GenAI spans on tool invocation:
   - "tool.execution.<name>" span with gen_ai.system, gen_ai.operation.name="execute_tool",
     gen_ai.tool.name, gen_ai.tool.call.id, gen_ai.tool.status="success" on successful execution.
   - gen_ai.tool.status="unregistered" on missing tool.
   - gen_ai.tool.status="unauthorized" on unauthorized tool.
   - gen_ai.tool.status="timeout" on timeout.
   - gen_ai.tool.status="error" and exception recording on tool failure.
5. AgentLoop emits structured GenAI spans:
   - "agent.plan_execution.<plan_id>" root span with gen_ai.plan.id, gen_ai.plan.step_count,
     gen_ai.plan.status.
   - "agent.step.<N>" child span for each step with gen_ai.step.number, gen_ai.step.action_type,
     gen_ai.step.rationale, and gen_ai.step.replayed flag.
6. Full trace hierarchy is properly formed and retrievable via in-memory span exporter.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.engine.agent_loop import AgentLoop, AgentPlanner
from src.engine.dispatcher import ParallaxToolDispatcher
from src.engine.executor import ExecutiveEngineProcess
from src.reasoning.schemas import ActionType, ReasoningStep, ToolCallPayload
from src.state.checkpoint_manager import (
    ConversationStateCheckpoint,
    compute_idempotency_key,
)
from src.telemetry import (
    AgentTelemetryTracer,
    GenAISpanAttributes,
    MockSpan,
    MockSpanExporter,
    StatusCode,
    record_span_exception,
    set_span_attribute,
    set_span_status,
    trace_agent_step,
)


class TestGenAISpanAttributes:
    """Verify GenAI semantic convention constants."""

    def test_semantic_convention_constants_format(self) -> None:
        """Verify attribute keys follow the gen_ai.* and error.* hierarchy."""
        assert GenAISpanAttributes.GEN_AI_SYSTEM == "gen_ai.system"
        assert GenAISpanAttributes.GEN_AI_OPERATION_NAME == "gen_ai.operation.name"
        assert GenAISpanAttributes.GEN_AI_AGENT_NAME == "gen_ai.agent.name"
        assert GenAISpanAttributes.GEN_AI_PLAN_ID == "gen_ai.plan.id"
        assert GenAISpanAttributes.GEN_AI_PLAN_STEP_COUNT == "gen_ai.plan.step_count"
        assert GenAISpanAttributes.GEN_AI_PLAN_STATUS == "gen_ai.plan.status"
        assert GenAISpanAttributes.GEN_AI_PLAN_DURATION_MS == "gen_ai.plan.duration_ms"
        assert GenAISpanAttributes.GEN_AI_STEP_NUMBER == "gen_ai.step.number"
        assert GenAISpanAttributes.GEN_AI_STEP_ACTION_TYPE == "gen_ai.step.action_type"
        assert GenAISpanAttributes.GEN_AI_STEP_RATIONALE == "gen_ai.step.rationale"
        assert GenAISpanAttributes.GEN_AI_STEP_IDEMPOTENCY_KEY == "gen_ai.step.idempotency_key"
        assert GenAISpanAttributes.GEN_AI_STEP_REPLAYED == "gen_ai.step.replayed"
        assert GenAISpanAttributes.GEN_AI_TOOL_NAME == "gen_ai.tool.name"
        assert GenAISpanAttributes.GEN_AI_TOOL_CALL_ID == "gen_ai.tool.call.id"
        assert GenAISpanAttributes.GEN_AI_TOOL_STATUS == "gen_ai.tool.status"
        assert GenAISpanAttributes.GEN_AI_TOOL_EXECUTION_TIME_MS == "gen_ai.tool.execution_time_ms"
        assert GenAISpanAttributes.GEN_AI_TOOL_TIMEOUT_SECONDS == "gen_ai.tool.timeout_seconds"
        assert GenAISpanAttributes.GEN_AI_TOOL_ARGUMENTS == "gen_ai.tool.arguments"
        assert GenAISpanAttributes.ERROR_TYPE == "error.type"
        assert GenAISpanAttributes.ERROR_MESSAGE == "error.message"


class TestMockSpanAndNormalization:
    """Verify MockSpan, MockSpanExporter, and attribute normalization."""

    def test_mock_span_lifecycle(self) -> None:
        exporter = MockSpanExporter()
        span = MockSpan("test_span", {"initial_key": "val1"}, exporter=exporter)
        with span as s:
            s.set_attribute("bool_key", True)
            s.set_attribute("int_key", 100)
            s.set_attribute("float_key", 3.14)
            s.set_attribute("dict_key", {"nested": "value"})
            s.set_status(StatusCode.OK)

        finished = exporter.get_finished_spans()
        assert len(finished) == 1
        recorded = finished[0]
        assert recorded.name == "test_span"
        assert recorded.attributes["initial_key"] == "val1"
        assert recorded.attributes["bool_key"] is True
        assert recorded.attributes["int_key"] == 100
        assert recorded.attributes["float_key"] == 3.14
        assert '"nested": "value"' in recorded.attributes["dict_key"]
        assert recorded.status == StatusCode.OK
        assert recorded.end_time is not None
        assert recorded.end_time >= recorded.start_time

    def test_mock_span_exception_recording(self) -> None:
        exporter = MockSpanExporter()
        span = MockSpan("test_error", exporter=exporter)
        err = ValueError("Test failure exception")
        with span as s:
            record_span_exception(s, err)

        finished = exporter.get_finished_spans()
        assert len(finished) == 1
        assert finished[0].status == StatusCode.ERROR
        assert "Test failure exception" in (finished[0].status_description or "")
        assert err in finished[0].exceptions

    def test_safe_helpers_with_none_span(self) -> None:
        """Verify helper functions do not crash when span is None."""
        set_span_attribute(None, "key", "value")
        set_span_status(None, StatusCode.OK)
        record_span_exception(None, RuntimeError("test"))

    def test_trace_agent_step_decorator(self) -> None:
        """Verify trace_agent_step decorator executes function and returns output."""

        @trace_agent_step("custom_decorated_step")
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5


class TestExecutiveEngineProcessSpans:
    """Verify ExecutiveEngineProcess emits GenAI spans across execution paths."""

    @pytest.mark.asyncio
    async def test_tool_execution_success_span(self) -> None:
        tracer_inst, exporter = AgentTelemetryTracer.create_in_memory_tracer("test-exec")
        executor = ExecutiveEngineProcess(tracer=tracer_inst)

        payload = ToolCallPayload(
            call_id="call-ping-1",
            tool_name="ping",
            arguments={"param": "hello"},
            timeout_seconds=5.0,
        )

        obs = await executor.execute_tool_call(payload)
        assert obs.success is True

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        attrs = dict(span.attributes)

        assert span.name == "tool.execution.ping"
        assert attrs[GenAISpanAttributes.GEN_AI_SYSTEM] == "athena-reasoning-sandbox"
        assert attrs[GenAISpanAttributes.GEN_AI_OPERATION_NAME] == "execute_tool"
        assert attrs[GenAISpanAttributes.GEN_AI_TOOL_NAME] == "ping"
        assert attrs[GenAISpanAttributes.GEN_AI_TOOL_CALL_ID] == "call-ping-1"
        assert attrs[GenAISpanAttributes.GEN_AI_TOOL_STATUS] == "success"
        assert GenAISpanAttributes.GEN_AI_TOOL_EXECUTION_TIME_MS in attrs

    @pytest.mark.asyncio
    async def test_tool_execution_unregistered_span(self) -> None:
        tracer_inst, exporter = AgentTelemetryTracer.create_in_memory_tracer("test-exec-unreg")
        executor = ExecutiveEngineProcess(tracer=tracer_inst)

        payload = ToolCallPayload(
            call_id="call-ghost",
            tool_name="non_existent_tool",
            arguments={},
            timeout_seconds=5.0,
        )

        obs = await executor.execute_tool_call(payload)
        assert obs.success is False

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)

        assert attrs[GenAISpanAttributes.GEN_AI_TOOL_NAME] == "non_existent_tool"
        assert attrs[GenAISpanAttributes.GEN_AI_TOOL_STATUS] == "unregistered"
        assert attrs[GenAISpanAttributes.ERROR_TYPE] == "UnregisteredToolError"
        assert "Unregistered executive tool" in attrs[GenAISpanAttributes.ERROR_MESSAGE]

    @pytest.mark.asyncio
    async def test_tool_execution_unauthorized_span(self) -> None:
        tracer_inst, exporter = AgentTelemetryTracer.create_in_memory_tracer("test-exec-unauth")
        executor = ExecutiveEngineProcess(authorized_tools={"echo"}, tracer=tracer_inst)

        payload = ToolCallPayload(
            call_id="call-ping-denied",
            tool_name="ping",
            arguments={},
            timeout_seconds=5.0,
        )

        obs = await executor.execute_tool_call(payload)
        assert obs.success is False

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)

        assert attrs[GenAISpanAttributes.GEN_AI_TOOL_STATUS] == "unauthorized"
        assert attrs[GenAISpanAttributes.ERROR_TYPE] == "UnauthorizedToolError"
        assert "not in the authorized tools allowlist" in attrs[GenAISpanAttributes.ERROR_MESSAGE]

    @pytest.mark.asyncio
    async def test_tool_execution_timeout_span(self) -> None:
        tracer_inst, exporter = AgentTelemetryTracer.create_in_memory_tracer("test-exec-timeout")
        executor = ExecutiveEngineProcess(tracer=tracer_inst)

        async def slow_handler(args: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(0.5)
            return {"status": "done"}

        executor.register_tool("slow_tool", slow_handler, authorize=True)

        payload = ToolCallPayload(
            call_id="call-slow",
            tool_name="slow_tool",
            arguments={},
            timeout_seconds=0.05,
        )

        obs = await executor.execute_tool_call(payload)
        assert obs.success is False
        assert "timed out" in (obs.error_message or "").lower()

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)

        assert attrs[GenAISpanAttributes.GEN_AI_TOOL_STATUS] == "timeout"
        assert attrs[GenAISpanAttributes.ERROR_TYPE] == "TimeoutError"
        assert "timed out" in attrs[GenAISpanAttributes.ERROR_MESSAGE].lower()

    @pytest.mark.asyncio
    async def test_tool_execution_exception_span(self) -> None:
        tracer_inst, exporter = AgentTelemetryTracer.create_in_memory_tracer("test-exec-err")
        executor = ExecutiveEngineProcess(tracer=tracer_inst)

        async def broken_handler(args: dict[str, Any]) -> dict[str, Any]:
            raise ArithmeticError("Division by zero in executive layer")

        executor.register_tool("broken_tool", broken_handler, authorize=True)

        payload = ToolCallPayload(
            call_id="call-broken",
            tool_name="broken_tool",
            arguments={},
            timeout_seconds=5.0,
        )

        obs = await executor.execute_tool_call(payload)
        assert obs.success is False

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        attrs = dict(span.attributes)

        assert attrs[GenAISpanAttributes.GEN_AI_TOOL_STATUS] == "error"
        assert attrs[GenAISpanAttributes.ERROR_TYPE] == "ArithmeticError"
        assert "Division by zero" in attrs[GenAISpanAttributes.ERROR_MESSAGE]


class TestAgentLoopSpans:
    """Verify AgentLoop emits GenAI plan and step spans."""

    @pytest.mark.asyncio
    async def test_successful_plan_spans(self) -> None:
        tracer_inst, exporter = AgentTelemetryTracer.create_in_memory_tracer("test-loop-success")
        executor = ExecutiveEngineProcess(tracer=tracer_inst)
        dispatcher = ParallaxToolDispatcher(executor=executor)
        loop = AgentLoop(dispatcher=dispatcher, tracer=tracer_inst)

        planner = AgentPlanner()
        steps = [
            ReasoningStep(
                step_number=1,
                rationale="Ping executive health",
                action_type=ActionType.SANDBOX_EXECUTE,
                tool_name="ping",
                tool_args={"host": "localhost"},
            ),
            ReasoningStep(
                step_number=2,
                rationale="Echo task status",
                action_type=ActionType.SANDBOX_EXECUTE,
                tool_name="echo",
                tool_args={"message": "complete"},
            ),
        ]
        plan = planner.create_plan(
            task_goal="Run 2 health checks", steps=steps, plan_id="plan-test-01"
        )

        result = await loop.run_plan(plan)
        assert result.success is True

        spans = exporter.get_finished_spans()
        # Expect 5 spans: tool ping, step 1, tool echo, step 2, plan_execution
        span_names = [s.name for s in spans]
        assert "agent.plan_execution.plan-test-01" in span_names
        assert "agent.step.1" in span_names
        assert "agent.step.2" in span_names
        assert "tool.execution.ping" in span_names
        assert "tool.execution.echo" in span_names

        # Check plan span attributes
        plan_span = next(s for s in spans if s.name == "agent.plan_execution.plan-test-01")
        plan_attrs = dict(plan_span.attributes)
        assert plan_attrs[GenAISpanAttributes.GEN_AI_SYSTEM] == "athena-reasoning-sandbox"
        assert plan_attrs[GenAISpanAttributes.GEN_AI_OPERATION_NAME] == "run_plan"
        assert plan_attrs[GenAISpanAttributes.GEN_AI_PLAN_ID] == "plan-test-01"
        assert plan_attrs[GenAISpanAttributes.GEN_AI_PLAN_STEP_COUNT] == 2
        assert plan_attrs[GenAISpanAttributes.GEN_AI_PLAN_STATUS] == "success"
        assert GenAISpanAttributes.GEN_AI_PLAN_DURATION_MS in plan_attrs

        # Check step span attributes
        step1_span = next(s for s in spans if s.name == "agent.step.1")
        step1_attrs = dict(step1_span.attributes)
        assert step1_attrs[GenAISpanAttributes.GEN_AI_STEP_NUMBER] == 1
        assert (
            step1_attrs[GenAISpanAttributes.GEN_AI_STEP_ACTION_TYPE]
            == ActionType.SANDBOX_EXECUTE.value
        )
        assert step1_attrs[GenAISpanAttributes.GEN_AI_TOOL_NAME] == "ping"
        assert step1_attrs[GenAISpanAttributes.GEN_AI_STEP_RATIONALE] == "Ping executive health"
        assert step1_attrs[GenAISpanAttributes.GEN_AI_STEP_REPLAYED] is False
        assert step1_attrs[GenAISpanAttributes.GEN_AI_TOOL_STATUS] == "success"

    @pytest.mark.asyncio
    async def test_replayed_step_span(self) -> None:
        tracer_inst, exporter = AgentTelemetryTracer.create_in_memory_tracer("test-loop-replay")
        executor = ExecutiveEngineProcess(tracer=tracer_inst)
        dispatcher = ParallaxToolDispatcher(executor=executor)
        loop = AgentLoop(dispatcher=dispatcher, tracer=tracer_inst)

        planner = AgentPlanner()
        step1 = ReasoningStep(
            step_number=1,
            rationale="Cached ping check",
            action_type=ActionType.SANDBOX_EXECUTE,
            tool_name="ping",
            tool_args={"cached": True},
        )
        plan = planner.create_plan(task_goal="Replay goal", steps=[step1], plan_id="plan-replay-01")

        # Pre-seed checkpoint ledger with step 1 effect
        action_type_str = step1.action_type.value
        key = compute_idempotency_key(action_type_str, step1.tool_name, step1.tool_args)
        checkpoint = ConversationStateCheckpoint(
            checkpoint_id="ck-replay",
            step_index=1,
        )
        checkpoint.record_effect(key, {"status": "replayed_pong"})

        result = await loop.run_plan(plan, checkpoint=checkpoint)
        assert result.success is True
        assert result.step_results[0].output_data == {"status": "replayed_pong"}

        spans = exporter.get_finished_spans()
        step_spans = [s for s in spans if s.name == "agent.step.1"]
        assert len(step_spans) == 1
        attrs = dict(step_spans[0].attributes)

        assert attrs[GenAISpanAttributes.GEN_AI_STEP_REPLAYED] is True
        assert attrs[GenAISpanAttributes.GEN_AI_TOOL_STATUS] == "replayed"
        # Since step was replayed, no tool span was emitted for ping
        tool_spans = [s for s in spans if s.name.startswith("tool.execution")]
        assert len(tool_spans) == 0

    @pytest.mark.asyncio
    async def test_failed_step_and_plan_span(self) -> None:
        tracer_inst, exporter = AgentTelemetryTracer.create_in_memory_tracer("test-loop-failure")
        executor = ExecutiveEngineProcess(authorized_tools=set(), tracer=tracer_inst)  # Deny all
        dispatcher = ParallaxToolDispatcher(executor=executor)
        loop = AgentLoop(dispatcher=dispatcher, tracer=tracer_inst)

        planner = AgentPlanner()
        step1 = ReasoningStep(
            step_number=1,
            rationale="Denied action",
            action_type=ActionType.SANDBOX_EXECUTE,
            tool_name="ping",
            tool_args={},
        )
        plan = planner.create_plan(
            task_goal="Expect failure", steps=[step1], plan_id="plan-fail-01"
        )

        result = await loop.run_plan(plan)
        assert result.success is False

        spans = exporter.get_finished_spans()
        plan_span = next(s for s in spans if s.name == "agent.plan_execution.plan-fail-01")
        plan_attrs = dict(plan_span.attributes)
        assert plan_attrs[GenAISpanAttributes.GEN_AI_PLAN_STATUS] == "failed"
        assert "Unauthorized tool execution" in plan_attrs[GenAISpanAttributes.ERROR_MESSAGE]

        step_span = next(s for s in spans if s.name == "agent.step.1")
        step_attrs = dict(step_span.attributes)
        assert step_attrs[GenAISpanAttributes.GEN_AI_TOOL_STATUS] == "error"
        assert "Unauthorized tool execution" in step_attrs[GenAISpanAttributes.ERROR_MESSAGE]
