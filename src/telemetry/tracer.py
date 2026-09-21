"""OpenTelemetry Distributed Tracing for Agentic Reasoning Trajectories.

Instruments the full reasoning loop:
Prompt -> Think -> Plan -> Tool Call -> Observation

Captures turn latency, token overhead, tool failure rate, and context length
using OpenTelemetry Semantic Conventions for Generative AI (v1.28+).

Algorithmic Complexity:
    - Span Creation & Attribute Injection: O(1) overhead per step.
"""

from __future__ import annotations

import functools
import json
import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Try importing opentelemetry; fallback gracefully if not installed
try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode

    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False
    trace = None  # type: ignore[assignment]

    class StatusCode:  # type: ignore[no-redef]
        """Fallback StatusCode when OpenTelemetry SDK is not installed."""

        UNSET = 0
        OK = 1
        ERROR = 2


class GenAISpanAttributes:
    """Standard semantic conventions for GenAI and Agent telemetry spans.

    Aligns with OpenTelemetry GenAI Semantic Conventions (v1.28+).
    """

    # System & Operations
    GEN_AI_SYSTEM = "gen_ai.system"
    GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
    GEN_AI_AGENT_NAME = "gen_ai.agent.name"

    # Plan-level attributes
    GEN_AI_PLAN_ID = "gen_ai.plan.id"
    GEN_AI_PLAN_STEP_COUNT = "gen_ai.plan.step_count"
    GEN_AI_PLAN_STATUS = "gen_ai.plan.status"
    GEN_AI_PLAN_DURATION_MS = "gen_ai.plan.duration_ms"

    # Step-level attributes
    GEN_AI_STEP_NUMBER = "gen_ai.step.number"
    GEN_AI_STEP_ACTION_TYPE = "gen_ai.step.action_type"
    GEN_AI_STEP_RATIONALE = "gen_ai.step.rationale"
    GEN_AI_STEP_IDEMPOTENCY_KEY = "gen_ai.step.idempotency_key"
    GEN_AI_STEP_REPLAYED = "gen_ai.step.replayed"

    # Tool-level attributes
    GEN_AI_TOOL_NAME = "gen_ai.tool.name"
    GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
    GEN_AI_TOOL_STATUS = "gen_ai.tool.status"
    GEN_AI_TOOL_EXECUTION_TIME_MS = "gen_ai.tool.execution_time_ms"
    GEN_AI_TOOL_TIMEOUT_SECONDS = "gen_ai.tool.timeout_seconds"
    GEN_AI_TOOL_ARGUMENTS = "gen_ai.tool.arguments"

    # Error attributes
    ERROR_TYPE = "error.type"
    ERROR_MESSAGE = "error.message"


def _normalize_attribute_value(value: Any) -> Any:
    """Normalize attribute values for OpenTelemetry compatibility.

    OpenTelemetry attributes must be primitive types (bool, str, int, float)
    or homogeneous sequences thereof. Complex dicts and objects are serialized to JSON.
    """
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(x, (bool, int, float, str)) for x in value
    ):
        return list(value)
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, default=str)
        except Exception:
            return str(value)
    return str(value)


def _normalize_attributes(attributes: dict[str, Any] | None) -> dict[str, Any] | None:
    """Filter and normalize attribute dictionary for OpenTelemetry span creation."""
    if not attributes:
        return None
    normalized: dict[str, Any] = {}
    for k, v in attributes.items():
        norm_val = _normalize_attribute_value(v)
        if norm_val is not None:
            normalized[k] = norm_val
    return normalized


def set_span_attribute(span: Any, key: str, value: Any) -> None:
    """Set an attribute on a span safely, normalizing complex types.

    Args:
        span: Active OpenTelemetry Span or MockSpan.
        key: Attribute key name.
        value: Attribute value to attach.
    """
    if span is None or value is None:
        return
    norm_val = _normalize_attribute_value(value)
    if norm_val is not None and hasattr(span, "set_attribute"):
        try:
            span.set_attribute(key, norm_val)
        except Exception as exc:
            logger.debug("Failed to set span attribute '%s': %s", key, exc)


def set_span_status(span: Any, status: Any, description: str | None = None) -> None:
    """Set the status of a span safely.

    Args:
        span: Active OpenTelemetry Span or MockSpan.
        status: StatusCode (OK, ERROR, UNSET) or status string.
        description: Optional human-readable description for errors.
    """
    if span is None:
        return
    if hasattr(span, "set_status"):
        try:
            span.set_status(status, description=description)
        except Exception as exc:
            logger.debug("Failed to set span status: %s", exc)


def record_span_exception(span: Any, exception: BaseException) -> None:
    """Record an exception on a span safely.

    Args:
        span: Active OpenTelemetry Span or MockSpan.
        exception: BaseException instance caught during execution.
    """
    if span is None:
        return
    if hasattr(span, "record_exception"):
        try:
            span.record_exception(exception)
        except Exception as exc:
            logger.debug("Failed to record exception on span: %s", exc)


class MockSpan:
    """Mock fallback span when OpenTelemetry SDK is uninstalled or during testing."""

    def __init__(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        exporter: MockSpanExporter | None = None,
    ) -> None:
        self.name = name
        self.attributes: dict[str, Any] = dict(attributes) if attributes else {}
        self.start_time = time.perf_counter()
        self.end_time: float | None = None
        self.status: Any = StatusCode.UNSET
        self.status_description: str | None = None
        self.exceptions: list[BaseException] = []
        self._exporter = exporter

    def __enter__(self) -> MockSpan:
        logger.debug("[OTel Span Start]: %s | Attrs: %s", self.name, self.attributes)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_time = time.perf_counter()
        elapsed_ms = (self.end_time - self.start_time) * 1000.0
        if exc_val is not None:
            self.status = StatusCode.ERROR
            self.status_description = str(exc_val)
            self.exceptions.append(exc_val)
        if self._exporter is not None:
            self._exporter.record_span(self)
        logger.debug("[OTel Span End]: %s | Latency: %.2fms", self.name, elapsed_ms)

    def set_attribute(self, key: str, value: Any) -> None:
        norm_val = _normalize_attribute_value(value)
        if norm_val is not None:
            self.attributes[key] = norm_val

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        for k, v in attributes.items():
            self.set_attribute(k, v)

    def set_status(self, status: Any, description: str | None = None) -> None:
        self.status = status
        self.status_description = description

    def record_exception(self, exception: BaseException) -> None:
        self.exceptions.append(exception)
        self.status = StatusCode.ERROR
        self.status_description = str(exception)


class MockSpanExporter:
    """In-memory collector for MockSpan objects used in tests without OpenTelemetry SDK."""

    def __init__(self) -> None:
        self._finished_spans: list[MockSpan] = []

    def record_span(self, span: MockSpan) -> None:
        self._finished_spans.append(span)

    def get_finished_spans(self) -> list[MockSpan]:
        return list(self._finished_spans)

    def clear(self) -> None:
        self._finished_spans.clear()


class AgentTelemetryTracer:
    """OpenTelemetry tracer manager for tracking agentic execution spans.

    Provides high-level span management with GenAI semantic conventions,
    gracefully falling back to in-memory mock spans when the SDK is absent.

    Attributes:
        service_name: Identifier for the traced service.
        tracer: OpenTelemetry Tracer instance if available.
    """

    def __init__(
        self,
        service_name: str = "athena-reasoning-sandbox",
        tracer: Any | None = None,
    ) -> None:
        """Initialize telemetry tracer.

        Args:
            service_name: Service identifier string.
            tracer: Optional explicit tracer instance (e.g. from custom TracerProvider).
        """
        self.service_name = service_name
        self.tracer: Any | None = None
        self._mock_exporter: MockSpanExporter | None = None

        if tracer is not None:
            self.tracer = tracer
        elif HAS_OPENTELEMETRY and trace:
            self.tracer = trace.get_tracer(service_name)
            logger.info("Initialized OpenTelemetry tracer for service '%s'", service_name)
        else:
            self._mock_exporter = MockSpanExporter()
            logger.info("OpenTelemetry SDK not installed. Running in mock fallback mode.")

    def start_span(self, name: str, attributes: dict[str, Any] | None = None) -> Any:
        """Start a new telemetry span for an agentic step or tool call.

        Args:
            name: Span name, e.g. "agent.plan_execution", "tool.execution.athena_query".
            attributes: Key-value attributes dict to attach.

        Returns:
            OpenTelemetry span context manager or MockSpan context manager.
        """
        norm_attrs = _normalize_attributes(attributes)
        if self.tracer is not None:
            return self.tracer.start_as_current_span(name, attributes=norm_attrs)
        else:
            return MockSpan(name, norm_attrs, exporter=self._mock_exporter)

    @classmethod
    def create_in_memory_tracer(
        cls,
        service_name: str = "athena-reasoning-sandbox",
    ) -> tuple[AgentTelemetryTracer, Any]:
        """Create an AgentTelemetryTracer backed by an in-memory exporter for testing.

        If OpenTelemetry SDK is installed, creates an isolated TracerProvider and
        InMemorySpanExporter. Otherwise, creates a MockSpanExporter.

        Args:
            service_name: Name of the test tracer.

        Returns:
            Tuple of (tracer_instance, in_memory_exporter).
        """
        if HAS_OPENTELEMETRY:
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
                InMemorySpanExporter,
            )

            provider = TracerProvider()
            exporter = InMemorySpanExporter()
            provider.add_span_processor(SimpleSpanProcessor(exporter))
            otel_tracer = provider.get_tracer(service_name)
            return cls(service_name=service_name, tracer=otel_tracer), exporter
        else:
            mock_tracer = cls(service_name=service_name)
            return mock_tracer, mock_tracer._mock_exporter  # type: ignore[return-value]


# Global tracer singleton instance
tracer = AgentTelemetryTracer()


def trace_agent_step(step_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to trace an agent step method automatically.

    Args:
        step_name: Name of the step span.

    Returns:
        Decorated function.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_span(step_name, {"function": func.__name__}):
                return func(*args, **kwargs)

        return wrapper

    return decorator
