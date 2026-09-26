"""Tests for ENG-01: retrieval_search tool handler registration.

Verifies that:
  - ``retrieval_search`` is in DEFAULT_AUTHORIZED_TOOLS.
  - ``ExecutiveEngineProcess`` registers the handler on construction.
  - The ``index`` operation indexes documents and returns counts.
  - The ``search`` operation returns ranked hits via BM25 RRF.
  - The ``clear`` operation resets the index to zero documents.
  - Missing ``query`` on search raises a ValueError through ObservationPayload.
  - Empty ``documents`` list on index raises a ValueError through ObservationPayload.
  - Searching an empty index raises a ValueError through ObservationPayload.
  - A pre-injected ``RetrievalIndex`` is respected (no lazy-init override).
  - The full end-to-end path through ``AgentLoop`` executes a
    RETRIEVAL_SEARCH plan step and returns a successful ObservationPayload.

All tests are Lane-1 (offline, deterministic, no external services).
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from src.engine.executor import DEFAULT_AUTHORIZED_TOOLS, ExecutiveEngineProcess
from src.rag.retrieval_index import RetrievalIndex
from src.reasoning.schemas import ActionType, ObservationPayload, ToolCallPayload

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def executor() -> ExecutiveEngineProcess:
    """Return a bare ExecutiveEngineProcess with no pre-injected clients."""
    return ExecutiveEngineProcess()


@pytest.fixture()
def seeded_index() -> RetrievalIndex:
    """Return a RetrievalIndex pre-loaded with three sample documents."""
    idx = RetrievalIndex()
    idx.index_documents([
        {"doc_id": "d1", "content": "Python is a high-level programming language."},
        {"doc_id": "d2", "content": "DuckDB is an in-process analytical database."},
        {"doc_id": "d3", "content": "Hybrid search combines BM25 and dense embeddings."},
    ])
    return idx


@pytest.fixture()
def executor_with_index(seeded_index: RetrievalIndex) -> ExecutiveEngineProcess:
    """Return an ExecutiveEngineProcess with a pre-seeded RetrievalIndex."""
    return ExecutiveEngineProcess(retrieval_index=seeded_index)


def _make_payload(args: dict, timeout: float = 10.0) -> ToolCallPayload:
    return ToolCallPayload(
        call_id=str(uuid.uuid4()),
        tool_name="retrieval_search",
        arguments=args,
        timeout_seconds=timeout,
    )


# ---------------------------------------------------------------------------
# ENG-01-A: Registration & authorisation
# ---------------------------------------------------------------------------

class TestEng01Registration:
    """Verify retrieval_search is registered and authorised by default."""

    def test_retrieval_search_in_default_authorized_tools(self) -> None:
        assert "retrieval_search" in DEFAULT_AUTHORIZED_TOOLS

    def test_retrieval_search_registered_on_construction(self, executor: ExecutiveEngineProcess) -> None:
        assert executor.is_authorized("retrieval_search")

    def test_registry_contains_retrieval_search(self, executor: ExecutiveEngineProcess) -> None:
        # Access private registry to confirm handler is wired
        assert "retrieval_search" in executor._registry  # noqa: SLF001


# ---------------------------------------------------------------------------
# ENG-01-B: Index operation
# ---------------------------------------------------------------------------

class TestEng01IndexOperation:
    """Verify the 'index' sub-operation correctly persists documents."""

    def test_index_returns_count(self, executor: ExecutiveEngineProcess) -> None:
        payload = _make_payload({
            "operation": "index",
            "documents": [
                {"doc_id": "a1", "content": "Alpha document about finance."},
                {"doc_id": "a2", "content": "Beta document about machine learning."},
            ],
        })
        result: ObservationPayload = asyncio.run(executor.execute_tool_call(payload))
        assert result.success is True
        assert result.output_data["operation"] == "index"
        assert result.output_data["indexed_count"] == 2
        assert result.output_data["total_docs"] == 2

    def test_index_accumulates_across_calls(self, executor: ExecutiveEngineProcess) -> None:
        asyncio.run(executor.execute_tool_call(_make_payload({
            "operation": "index",
            "documents": [{"doc_id": "b1", "content": "First batch."}],
        })))
        result: ObservationPayload = asyncio.run(executor.execute_tool_call(_make_payload({
            "operation": "index",
            "documents": [{"doc_id": "b2", "content": "Second batch."}],
        })))
        assert result.success is True
        assert result.output_data["total_docs"] == 2

    def test_index_empty_documents_returns_error(self, executor: ExecutiveEngineProcess) -> None:
        payload = _make_payload({"operation": "index", "documents": []})
        result: ObservationPayload = asyncio.run(executor.execute_tool_call(payload))
        assert result.success is False
        assert "requires a non-empty" in (result.error_message or "")

    def test_index_missing_documents_key_returns_error(self, executor: ExecutiveEngineProcess) -> None:
        payload = _make_payload({"operation": "index"})
        result: ObservationPayload = asyncio.run(executor.execute_tool_call(payload))
        assert result.success is False
        assert "requires a non-empty" in (result.error_message or "")

    def test_index_document_missing_content_raises(self, executor: ExecutiveEngineProcess) -> None:
        payload = _make_payload({
            "operation": "index",
            "documents": [{"doc_id": "bad_doc"}],  # missing 'content'
        })
        result: ObservationPayload = asyncio.run(executor.execute_tool_call(payload))
        assert result.success is False
        assert "content" in (result.error_message or "")


# ---------------------------------------------------------------------------
# ENG-01-C: Search operation
# ---------------------------------------------------------------------------

class TestEng01SearchOperation:
    """Verify the 'search' (default) sub-operation returns ranked hits."""

    def test_search_returns_hits(self, executor_with_index: ExecutiveEngineProcess) -> None:
        payload = _make_payload({"query": "database analytics"})
        result: ObservationPayload = asyncio.run(executor_with_index.execute_tool_call(payload))
        assert result.success is True
        data = result.output_data
        assert "total_hits" in data
        assert "hits" in data
        assert len(data["hits"]) > 0

    def test_search_hit_structure(self, executor_with_index: ExecutiveEngineProcess) -> None:
        payload = _make_payload({"query": "programming"})
        result: ObservationPayload = asyncio.run(executor_with_index.execute_tool_call(payload))
        assert result.success is True
        hit = result.output_data["hits"][0]
        assert "doc_id" in hit
        assert "content" in hit
        assert "rrf_score" in hit
        assert "bm25_rank" in hit
        assert "dense_rank" in hit

    def test_search_respects_top_k(self, executor_with_index: ExecutiveEngineProcess) -> None:
        payload = _make_payload({"query": "search", "top_k": 2})
        result: ObservationPayload = asyncio.run(executor_with_index.execute_tool_call(payload))
        assert result.success is True
        assert len(result.output_data["hits"]) <= 2

    def test_search_with_dense_embedding(self, executor_with_index: ExecutiveEngineProcess) -> None:
        # Provide a simple unit-vector embedding (dimension=3 — must not match doc embeddings
        # so dense score is 0, but BM25 still produces a result).
        payload = _make_payload({
            "query": "hybrid search",
            "query_embedding": [0.1, 0.9, 0.0],
            "top_k": 3,
        })
        result: ObservationPayload = asyncio.run(executor_with_index.execute_tool_call(payload))
        assert result.success is True
        assert result.output_data["total_hits"] > 0

    def test_search_empty_query_returns_error(self, executor_with_index: ExecutiveEngineProcess) -> None:
        payload = _make_payload({"query": ""})
        result: ObservationPayload = asyncio.run(executor_with_index.execute_tool_call(payload))
        assert result.success is False
        assert "requires a non-empty 'query'" in (result.error_message or "")

    def test_search_missing_query_returns_error(self, executor_with_index: ExecutiveEngineProcess) -> None:
        payload = _make_payload({})
        result: ObservationPayload = asyncio.run(executor_with_index.execute_tool_call(payload))
        assert result.success is False

    def test_search_empty_index_returns_error(self, executor: ExecutiveEngineProcess) -> None:
        """Searching before any documents are indexed should fail gracefully."""
        payload = _make_payload({"query": "anything"})
        result: ObservationPayload = asyncio.run(executor.execute_tool_call(payload))
        assert result.success is False
        assert "empty" in (result.error_message or "").lower()

    def test_explicit_search_operation_key(self, executor_with_index: ExecutiveEngineProcess) -> None:
        payload = _make_payload({"operation": "search", "query": "analytics"})
        result: ObservationPayload = asyncio.run(executor_with_index.execute_tool_call(payload))
        assert result.success is True


# ---------------------------------------------------------------------------
# ENG-01-D: Clear operation
# ---------------------------------------------------------------------------

class TestEng01ClearOperation:
    """Verify the 'clear' sub-operation resets the index."""

    def test_clear_resets_document_count(self, executor_with_index: ExecutiveEngineProcess) -> None:
        result: ObservationPayload = asyncio.run(executor_with_index.execute_tool_call(
            _make_payload({"operation": "clear"})
        ))
        assert result.success is True
        assert result.output_data["total_docs"] == 0

    def test_search_after_clear_returns_error(self, executor_with_index: ExecutiveEngineProcess) -> None:
        asyncio.run(executor_with_index.execute_tool_call(_make_payload({"operation": "clear"})))
        result: ObservationPayload = asyncio.run(executor_with_index.execute_tool_call(
            _make_payload({"query": "anything"})
        ))
        assert result.success is False
        assert "empty" in (result.error_message or "").lower()


# ---------------------------------------------------------------------------
# ENG-01-E: Pre-injected index is respected
# ---------------------------------------------------------------------------

class TestEng01InjectedIndex:
    """Verify that a custom RetrievalIndex injected at construction is used."""

    def test_injected_index_is_not_replaced(self, seeded_index: RetrievalIndex) -> None:
        exe = ExecutiveEngineProcess(retrieval_index=seeded_index)
        assert exe._retrieval_index is seeded_index  # noqa: SLF001

    def test_search_uses_injected_index_documents(self, executor_with_index: ExecutiveEngineProcess) -> None:
        payload = _make_payload({"query": "DuckDB analytical"})
        result: ObservationPayload = asyncio.run(executor_with_index.execute_tool_call(payload))
        assert result.success is True
        doc_ids = [h["doc_id"] for h in result.output_data["hits"]]
        assert "d2" in doc_ids  # "DuckDB is an in-process analytical database."


# ---------------------------------------------------------------------------
# ENG-01-F: End-to-end AgentLoop integration
# ---------------------------------------------------------------------------

class TestEng01AgentLoopIntegration:
    """End-to-end: RETRIEVAL_SEARCH plan step dispatched through the full Parallax boundary."""

    def test_agent_loop_executes_retrieval_search_step(self, executor_with_index: ExecutiveEngineProcess) -> None:
        from src.engine.agent_loop import AgentLoop, AgentPlanner
        from src.engine.dispatcher import ParallaxToolDispatcher
        from src.reasoning.schemas import ReasoningStep

        dispatcher = ParallaxToolDispatcher(executor=executor_with_index)
        loop = AgentLoop(dispatcher=dispatcher)
        planner = AgentPlanner()

        step = ReasoningStep(
            step_number=1,
            rationale="Retrieve relevant documents about Python programming.",
            action_type=ActionType.RETRIEVAL_SEARCH,
            tool_name="retrieval_search",
            tool_args={"operation": "search", "query": "Python programming language", "top_k": 3},
        )
        plan = planner.create_plan(
            task_goal="Find documents about Python.",
            steps=[step],
            plan_id=str(uuid.uuid4()),
        )
        loop_result = asyncio.run(loop.run_plan(plan=plan))

        assert loop_result.success is True
        assert loop_result.final_output is not None
        hits = loop_result.final_output.get("hits", [])
        assert len(hits) > 0
        # d1 ("Python is a high-level programming language.") must be top-ranked
        assert hits[0]["doc_id"] == "d1"


# ---------------------------------------------------------------------------
# Metadata Filtering Integration Tests
# ---------------------------------------------------------------------------

class TestRetrievalSearchMetadataFiltering:
    """Verifies metadata-based candidate pre-filtering in Hybrid Search and Executive Engine."""

    def test_search_with_matching_metadata_filter(self) -> None:
        idx = RetrievalIndex()
        idx.index_documents([
            {"doc_id": "doc1", "content": "Athena SQL cloud billing", "metadata": {"category": "finops"}},
            {"doc_id": "doc2", "content": "DuckDB SQL in-memory execution", "metadata": {"category": "database"}},
            {"doc_id": "doc3", "content": "AWS Athena data lake querying", "metadata": {"category": "finops"}},
        ])

        results = idx.search("SQL querying", filter_metadata={"category": "finops"}, top_k=5)
        assert len(results) == 2
        doc_ids = {r.document.doc_id for r in results}
        assert doc_ids == {"doc1", "doc3"}
        assert "doc2" not in doc_ids

    def test_search_with_non_matching_metadata_filter(self) -> None:
        idx = RetrievalIndex()
        idx.index_documents([
            {"doc_id": "doc1", "content": "Athena SQL cloud billing", "metadata": {"category": "finops"}},
        ])

        results = idx.search("Athena", filter_metadata={"category": "machine_learning"}, top_k=5)
        assert len(results) == 0

    def test_executor_retrieval_search_with_filter_metadata(self) -> None:
        idx = RetrievalIndex()
        idx.index_documents([
            {"doc_id": "doc1", "content": "Machine learning model weights", "metadata": {"tag": "ai"}},
            {"doc_id": "doc2", "content": "FinOps cloud spend management", "metadata": {"tag": "finops"}},
        ])
        exe = ExecutiveEngineProcess(retrieval_index=idx)
        payload = _make_payload({
            "operation": "search",
            "query": "management and models",
            "filter_metadata": {"tag": "finops"},
        })
        result: ObservationPayload = asyncio.run(exe.execute_tool_call(payload))
        assert result.success is True
        hits = result.output_data["hits"]
        assert len(hits) == 1
        assert hits[0]["doc_id"] == "doc2"

