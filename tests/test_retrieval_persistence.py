"""Unit and Integration Tests for Retrieval Index Persistence and Operations.

Validates:
1. Document upsert behavior (updating existing doc_id vs skipping when upsert=False).
2. Document lookup by ID (get_document).
3. Document deletion and search engine index rebuild (delete_document).
4. Full index document export (export_documents).
5. JSON disk persistence (save_to_json and load_from_json).
6. Tool dispatch via ExecutiveEngineProcess across all operations.
7. MCP JSON-RPC tool dispatch for retrieval_search.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.engine.executor import ExecutiveEngineProcess
from src.engine.mcp_server import AthenaMCPServer
from src.rag.retrieval_index import RetrievalIndex
from src.reasoning.schemas import ToolCallPayload


@pytest.fixture
def sample_docs() -> list[dict[str, str]]:
    return [
        {
            "doc_id": "doc-1",
            "content": "DuckDB is an in-process analytical SQL database designed for OLAP.",
        },
        {
            "doc_id": "doc-2",
            "content": "Amazon Athena is a serverless interactive analytics service using Presto/Trino.",
        },
        {
            "doc_id": "doc-3",
            "content": "Model Context Protocol connects AI assistants to local development tools.",
        },
    ]


class TestRetrievalIndexCoreOperations:
    """Tests for core document management in RetrievalIndex."""

    def test_upsert_document_updates_content(self, sample_docs: list[dict[str, str]]) -> None:
        index = RetrievalIndex()
        index.index_documents(sample_docs)
        assert index.document_count == 3

        # Upsert doc-1 with new content
        index.index_documents([
            {"doc_id": "doc-1", "content": "DuckDB supports columnar vectorised query execution."}
        ])
        assert index.document_count == 3
        doc = index.get_document("doc-1")
        assert doc is not None
        assert "columnar vectorised" in doc["content"]

        # Search should reflect the updated document content
        results = index.search(query="columnar vectorised")
        assert len(results) > 0
        assert results[0].document.doc_id == "doc-1"

    def test_no_upsert_skips_duplicate_doc_id(self, sample_docs: list[dict[str, str]]) -> None:
        index = RetrievalIndex()
        index.index_documents(sample_docs)

        # Attempt to insert doc-1 with upsert=False
        modified = index.index_documents(
            [{"doc_id": "doc-1", "content": "Different content that should be skipped"}],
            upsert=False,
        )
        assert modified == 0
        doc = index.get_document("doc-1")
        assert doc is not None
        assert "OLAP" in doc["content"]

    def test_get_document_nonexistent_returns_none(self) -> None:
        index = RetrievalIndex()
        assert index.get_document("missing-doc") is None

    def test_delete_document_removes_from_search(self, sample_docs: list[dict[str, str]]) -> None:
        index = RetrievalIndex()
        index.index_documents(sample_docs)
        assert index.document_count == 3

        deleted = index.delete_document("doc-2")
        assert deleted is True
        assert index.document_count == 2
        assert index.get_document("doc-2") is None

        # Deleting non-existent returns False
        assert index.delete_document("doc-2") is False

        # Search for deleted document terms should not return doc-2
        results = index.search(query="Athena serverless interactive")
        for res in results:
            assert res.document.doc_id != "doc-2"

    def test_export_documents_format(self, sample_docs: list[dict[str, str]]) -> None:
        index = RetrievalIndex()
        index.index_documents(sample_docs)
        exported = index.export_documents()
        assert len(exported) == 3
        doc_ids = {d["doc_id"] for d in exported}
        assert doc_ids == {"doc-1", "doc-2", "doc-3"}


class TestRetrievalIndexDiskPersistence:
    """Tests for saving and loading RetrievalIndex from JSON files."""

    def test_save_and_load_roundtrip(self, sample_docs: list[dict[str, str]], tmp_path: Path) -> None:
        index = RetrievalIndex()
        index.index_documents(sample_docs)

        filepath = tmp_path / "subfolder" / "index.json"
        index.save_to_json(filepath)
        assert filepath.is_file()

        with open(filepath, encoding="utf-8") as f:
            saved_data = json.load(f)
        assert len(saved_data) == 3

        # Load into a fresh index
        new_index = RetrievalIndex()
        loaded_count = new_index.load_from_json(filepath)
        assert loaded_count == 3
        assert new_index.document_count == 3

        # Search works on loaded index
        results = new_index.search(query="Model Context Protocol")
        assert len(results) > 0
        assert results[0].document.doc_id == "doc-3"

    def test_load_from_json_with_clear_existing(self, sample_docs: list[dict[str, str]], tmp_path: Path) -> None:
        filepath = tmp_path / "saved.json"
        index1 = RetrievalIndex()
        index1.index_documents([sample_docs[0]])
        index1.save_to_json(filepath)

        index2 = RetrievalIndex()
        index2.index_documents([sample_docs[1], sample_docs[2]])
        assert index2.document_count == 2

        # Load with clear_existing=True
        index2.load_from_json(filepath, clear_existing=True)
        assert index2.document_count == 1
        assert index2.get_document("doc-1") is not None
        assert index2.get_document("doc-2") is None


class TestRetrievalExecutorToolIntegration:
    """Tests for ExecutiveEngineProcess retrieval_search tool handling."""

    @pytest.mark.asyncio
    async def test_executor_save_load_and_export_operations(self, tmp_path: Path) -> None:
        index = RetrievalIndex()
        executor = ExecutiveEngineProcess(retrieval_index=index)
        save_file = str(tmp_path / "executor_index.json")

        # 1. Index documents
        obs_index = await executor.execute_tool_call(
            ToolCallPayload(
                call_id="call-1",
                tool_name="retrieval_search",
                arguments={
                    "operation": "index",
                    "documents": [
                        {"doc_id": "kb-1", "content": "Fine-tuning QLoRA memory optimization."},
                        {"doc_id": "kb-2", "content": "SLERP parameter merging in hyperspherical space."},
                    ],
                },
            )
        )
        assert obs_index.success is True
        assert obs_index.output_data["total_docs"] == 2

        # 2. Get document
        obs_get = await executor.execute_tool_call(
            ToolCallPayload(
                call_id="call-2",
                tool_name="retrieval_search",
                arguments={"operation": "get", "doc_id": "kb-1"},
            )
        )
        assert obs_get.success is True
        assert obs_get.output_data["document"]["doc_id"] == "kb-1"

        # 3. Save to disk
        obs_save = await executor.execute_tool_call(
            ToolCallPayload(
                call_id="call-3",
                tool_name="retrieval_search",
                arguments={"operation": "save", "filepath": save_file},
            )
        )
        assert obs_save.success is True
        assert Path(save_file).is_file()

        # 4. Clear index
        obs_clear = await executor.execute_tool_call(
            ToolCallPayload(
                call_id="call-4",
                tool_name="retrieval_search",
                arguments={"operation": "clear"},
            )
        )
        assert obs_clear.success is True
        assert obs_clear.output_data["total_docs"] == 0

        # 5. Load from disk
        obs_load = await executor.execute_tool_call(
            ToolCallPayload(
                call_id="call-5",
                tool_name="retrieval_search",
                arguments={"operation": "load", "filepath": save_file},
            )
        )
        assert obs_load.success is True
        assert obs_load.output_data["total_docs"] == 2

        # 6. Delete single doc
        obs_del = await executor.execute_tool_call(
            ToolCallPayload(
                call_id="call-6",
                tool_name="retrieval_search",
                arguments={"op": "delete", "doc_id": "kb-2"},
            )
        )
        assert obs_del.success is True
        assert obs_del.output_data["total_docs"] == 1


class TestRetrievalMCPServerIntegration:
    """Tests for retrieval_search over MCP JSON-RPC protocol."""

    @pytest.mark.asyncio
    async def test_mcp_retrieval_search_lifecycle(self) -> None:
        index = RetrievalIndex()
        executor = ExecutiveEngineProcess(retrieval_index=index)
        server = AthenaMCPServer(executor=executor)

        # 1. MCP tool call: index
        req_index = {
            "jsonrpc": "2.0",
            "id": "mcp-req-1",
            "method": "tools/call",
            "params": {
                "name": "retrieval_search",
                "arguments": {
                    "op": "index",
                    "documents": [
                        {"doc_id": "doc-a", "content": "Distributed data processing with PySpark."},
                        {"doc_id": "doc-b", "content": "Analytical query pruning with Athena and DuckDB."},
                    ],
                },
            },
        }
        res_index = await server.handle_message(req_index)
        assert res_index is not None
        assert "result" in res_index
        assert res_index["result"]["isError"] is False

        # 2. MCP tool call: search
        req_search = {
            "jsonrpc": "2.0",
            "id": "mcp-req-2",
            "method": "tools/call",
            "params": {
                "name": "retrieval_search",
                "arguments": {
                    "op": "search",
                    "query": "DuckDB analytical query pruning",
                    "top_k": 2,
                },
            },
        }
        res_search = await server.handle_message(req_search)
        assert res_search is not None
        assert res_search["result"]["isError"] is False
        text_content = res_search["result"]["content"][0]["text"]
        assert "doc-b" in text_content
