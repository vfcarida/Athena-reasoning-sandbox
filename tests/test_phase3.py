"""Unit test suite for Phase 3: MLOps DeepEval & Observability.

Verifies custom trajectory evaluation metrics, Hybrid Search with Reciprocal Rank Fusion,
and OpenTelemetry tracer span creation.
"""

import pytest
from deepeval.test_case import LLMTestCase
from evals.metrics.trajectory_metrics import (
    PlanQualityMetric,
    PlanAdherenceMetric,
    ToolCorrectnessMetric,
    TaskCompletionMetric,
)
from src.rag.hybrid_search import HybridSearchEngine, Document
from src.telemetry.tracer import AgentTelemetryTracer


def test_plan_quality_metric():
    """Verify PlanQualityMetric evaluates structured reasoning plans."""
    metric = PlanQualityMetric(threshold=0.75)
    tc = LLMTestCase(
        input="Analyze dataset partitions",
        actual_output="Step 1: Inspect schema. Step 2: Validate partition column dt.",
    )
    score = metric.measure(tc)
    assert score >= 0.75
    assert metric.is_successful()


def test_plan_adherence_metric():
    """Verify PlanAdherenceMetric evaluates plan step execution matches."""
    metric = PlanAdherenceMetric(threshold=0.75)
    tc = LLMTestCase(
        input="Query logs",
        actual_output="Execute step 1 query logs where dt = '2026-08-12'",
        retrieval_context=["Execute step 1 query logs where dt = '2026-08-12'"],
    )
    score = metric.measure(tc)
    assert score == 1.0
    assert metric.is_successful()


def test_tool_correctness_metric():
    """Verify ToolCorrectnessMetric evaluates tool handle invocation."""
    metric = ToolCorrectnessMetric(threshold=0.75)
    tc = LLMTestCase(
        input="Run ping test",
        actual_output="Executing ping tool call",
    )
    score = metric.measure(tc)
    assert score >= 0.75


def test_task_completion_metric():
    """Verify TaskCompletionMetric evaluates non-empty execution output."""
    metric = TaskCompletionMetric(threshold=0.75)
    tc = LLMTestCase(input="Task goal", actual_output="Success result text")
    score = metric.measure(tc)
    assert score == 0.90


def test_hybrid_search_rrf():
    """Verify HybridSearchEngine indexes documents and ranks via RRF."""
    engine = HybridSearchEngine(rrf_k=60)
    docs = [
        Document(doc_id="doc1", content="Athena SQL queries must filter partition keys like dt.", embedding=[0.1, 0.2, 0.3]),
        Document(doc_id="doc2", content="SwiReasoning calculates Shannon entropy for logits.", embedding=[0.9, 0.1, 0.0]),
    ]
    engine.index_documents(docs)

    results = engine.search(query="Athena SQL partition keys", query_embedding=[0.1, 0.2, 0.3], top_k=2)
    assert len(results) == 2
    assert results[0].document.doc_id == "doc1"
    assert results[0].rrf_score > 0.0


def test_telemetry_tracer_spans():
    """Verify AgentTelemetryTracer creates spans without errors."""
    tracer = AgentTelemetryTracer()
    with tracer.start_span("Prompt", {"prompt_length": 120}):
        pass
    with tracer.start_span("Think", {"entropy": 2.1}):
        pass
