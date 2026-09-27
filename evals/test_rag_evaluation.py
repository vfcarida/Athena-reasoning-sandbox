"""Deterministic RAG Evaluation Test Suite.

Evaluates RAG retrieval and generation against honest deterministic metrics (Lane 1, offline):
- DeterministicContextRelevancyMetric: Evaluates query keyword density in retrieved chunks.
- DeterministicContextPrecisionMetric: Evaluates rank-weighted positioning (MAP / Precision@k).
- DeterministicFaithfulnessMetric: Evaluates groundedness of output statements against context.
- DeterministicRAGGate: Composite gate combining relevancy, precision, and faithfulness.

All tests run completely offline without paid external LLM judge APIs.
"""

from __future__ import annotations

import pytest
from deepeval import assert_test  # type: ignore[attr-defined]
from deepeval.test_case import LLMTestCase

from evals.metrics.rag_metrics import (
    DeterministicContextPrecisionMetric,
    DeterministicContextRelevancyMetric,
    DeterministicFaithfulnessMetric,
    DeterministicRAGGate,
)


@pytest.fixture()
def good_rag_test_case() -> LLMTestCase:
    """Return an LLMTestCase with high relevancy, top-ranked precision, and faithful output."""
    return LLMTestCase(
        input="What is the pricing model and partition pruning strategy for AWS Athena?",
        actual_output=(
            "AWS Athena charges 5.00 dollars per terabyte scanned. "
            "To reduce query cost, engineers apply partition pruning on partition keys like dt. "
            "Partition pruning ensures Athena scans only relevant S3 prefix directories."
        ),
        retrieval_context=[
            "AWS Athena charges $5.00 per terabyte of data scanned from Amazon S3.",
            "Partition pruning restricts scans by evaluating partition keys such as dt = '2026-09-01'.",
            "Queries without partition keys trigger full Athena scans across all S3 prefix directories, failing to prune data.",
        ],
    )



@pytest.fixture()
def irrelevant_rag_test_case() -> LLMTestCase:
    """Return an LLMTestCase where retrieved contexts do not match query keywords."""
    return LLMTestCase(
        input="What is the pricing model and partition pruning strategy for AWS Athena?",
        actual_output="Athena scans data efficiently.",
        retrieval_context=[
            "Spherical linear interpolation SLERP merges model weights along high-dimensional hyperspheres.",
            "DARE drops delta parameters randomly and rescales remaining weights.",
        ],
    )


@pytest.fixture()
def unfaithful_rag_test_case() -> LLMTestCase:
    """Return an LLMTestCase where output contains hallucinated claims not in context."""
    return LLMTestCase(
        input="What is the pricing model for AWS Athena?",
        actual_output=(
            "AWS Athena costs 100 dollars per minute of compute. "
            "Athena automatically compiles queries to GPU CUDA kernels running on H100 clusters."
        ),
        retrieval_context=[
            "AWS Athena charges $5.00 per terabyte of data scanned from Amazon S3.",
            "Athena is a serverless interactive query service based on Presto/Trino.",
        ],
    )


def test_context_relevancy_metric_passes_on_good_case(good_rag_test_case: LLMTestCase) -> None:
    """Verify DeterministicContextRelevancyMetric scores high on relevant chunks."""
    metric = DeterministicContextRelevancyMetric(threshold=0.70)
    score = metric.measure(good_rag_test_case)

    assert score >= 0.70, f"Expected score >= 0.70, got {score}: {metric.reason}"
    assert metric.is_successful() is True
    assert metric.details["relevant_count"] == 3


def test_context_relevancy_metric_fails_on_irrelevant_chunks(irrelevant_rag_test_case: LLMTestCase) -> None:
    """Verify DeterministicContextRelevancyMetric fails on off-topic contexts."""
    metric = DeterministicContextRelevancyMetric(threshold=0.70)
    score = metric.measure(irrelevant_rag_test_case)

    assert score < 0.50, f"Expected low score for off-topic context, got {score}"
    assert metric.is_successful() is False


def test_context_precision_metric_favors_top_ranked_context() -> None:
    """Verify Precision@k scores higher when relevant chunks appear first."""
    # Top-ranked relevant chunk
    top_ranked_case = LLMTestCase(
        input="DuckDB analytical engine performance",
        actual_output="DuckDB executes columnar queries in-process with vectorized execution.",
        retrieval_context=[
            "DuckDB is an in-process columnar analytical query engine designed for high performance.",
            "Random unrelated sentence about weather and climate in California.",
        ],
    )

    # Bottom-ranked relevant chunk
    bottom_ranked_case = LLMTestCase(
        input="DuckDB analytical engine performance",
        actual_output="DuckDB executes columnar queries in-process with vectorized execution.",
        retrieval_context=[
            "Random unrelated sentence about weather and climate in California.",
            "DuckDB is an in-process columnar analytical query engine designed for high performance.",
        ],
    )

    metric = DeterministicContextPrecisionMetric(threshold=0.50)
    top_score = metric.measure(top_ranked_case)
    bottom_score = metric.measure(bottom_ranked_case)

    assert top_score > bottom_score, f"Top ranked ({top_score}) should beat bottom ranked ({bottom_score})"
    assert top_score == 1.0  # Relevant at k=1 -> Precision@1 = 1.0


def test_faithfulness_metric_passes_on_grounded_output(good_rag_test_case: LLMTestCase) -> None:
    """Verify grounded output passes DeterministicFaithfulnessMetric."""
    metric = DeterministicFaithfulnessMetric(threshold=0.70)
    score = metric.measure(good_rag_test_case)

    assert score >= 0.70, f"Expected faithfulness >= 0.70, got {score}: {metric.reason}"
    assert metric.is_successful() is True


def test_faithfulness_metric_fails_on_hallucinations(unfaithful_rag_test_case: LLMTestCase) -> None:
    """Verify hallucinated output fails DeterministicFaithfulnessMetric."""
    metric = DeterministicFaithfulnessMetric(threshold=0.70)
    score = metric.measure(unfaithful_rag_test_case)

    assert score < 0.60, f"Expected low faithfulness score for hallucinations, got {score}"
    assert metric.is_successful() is False


def test_deterministic_rag_gate_composite(good_rag_test_case: LLMTestCase) -> None:
    """Verify DeterministicRAGGate validates high-quality retrieval and generation."""
    gate = DeterministicRAGGate(threshold=0.70)
    score = gate.measure(good_rag_test_case)

    assert score >= 0.70, f"Expected gate >= 0.70, got {score}: {gate.reason}"
    assert gate.is_successful() is True
    assert "relevancy" in gate.details
    assert "precision" in gate.details
    assert "faithfulness" in gate.details

    # Assert test integration with deepeval
    assert_test(test_case=good_rag_test_case, metrics=[gate])


def test_rag_metrics_edge_cases_empty_input() -> None:
    """Verify graceful handling of empty inputs and contexts."""
    empty_case = LLMTestCase(input="", actual_output="", retrieval_context=[])

    rel = DeterministicContextRelevancyMetric()
    prec = DeterministicContextPrecisionMetric()
    faith = DeterministicFaithfulnessMetric()
    gate = DeterministicRAGGate()

    assert rel.measure(empty_case) == 0.0
    assert prec.measure(empty_case) == 0.0
    assert faith.measure(empty_case) == 0.0
    assert gate.measure(empty_case) == 0.0
    assert gate.is_successful() is False
