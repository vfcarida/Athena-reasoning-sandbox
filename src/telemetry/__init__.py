"""OpenTelemetry telemetry and distributed tracing module."""

from src.telemetry.tracer import (
    AgentTelemetryTracer,
    GenAISpanAttributes,
    MockSpan,
    MockSpanExporter,
    StatusCode,
    record_span_exception,
    set_span_attribute,
    set_span_status,
    trace_agent_step,
    tracer,
)

__all__ = [
    "AgentTelemetryTracer",
    "GenAISpanAttributes",
    "MockSpan",
    "MockSpanExporter",
    "StatusCode",
    "record_span_exception",
    "set_span_attribute",
    "set_span_status",
    "trace_agent_step",
    "tracer",
]
