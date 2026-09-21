"""In-Process Retrieval Index — Stateful document store for the retrieval_search tool.

Provides a process-level singleton index wrapping ``HybridSearchEngine`` so that the
``ExecutiveEngineProcess`` tool handler can index, query, and clear documents without
requiring any external vector-database service.

This is the Lane-1 (offline) implementation.  Production deployments may substitute
a remote backend (e.g. Pinecone, Weaviate) by providing a custom ``RetrievalBackend``
implementation and injecting it into ``ExecutiveEngineProcess``.

Algorithmic Complexity:
    - ``index()``: O(D) where D is the number of new documents.
    - ``search()``: O(N * Q) where N is the total indexed docs and Q is the query length
      (BM25 dominant term); dense scoring adds O(N * V) where V is embedding dimension.
"""

from __future__ import annotations

import logging
from typing import Any

from src.rag.hybrid_search import Document, HybridSearchEngine, SearchResult

logger = logging.getLogger(__name__)


class RetrievalIndex:
    """Stateful in-process document retrieval index backed by ``HybridSearchEngine``.

    Designed to be injected into ``ExecutiveEngineProcess`` so that the
    ``retrieval_search`` tool handler has access to a live, query-able document store.

    Attributes:
        _engine: Underlying ``HybridSearchEngine`` instance.
        _doc_count: Running total of documents indexed across all ``index_documents`` calls.
    """

    def __init__(self, rrf_k: int = 60) -> None:
        """Initialise the retrieval index.

        Args:
            rrf_k: Reciprocal Rank Fusion smoothing constant (default 60).
        """
        self._engine = HybridSearchEngine(rrf_k=rrf_k)
        self._doc_count: int = 0

    @property
    def document_count(self) -> int:
        """Return total number of indexed documents."""
        return self._doc_count

    def index_documents(self, documents: list[dict[str, Any]]) -> int:
        """Add documents to the index.

        Each document dict must contain ``doc_id`` and ``content`` keys.
        Optional keys: ``metadata`` (dict) and ``embedding`` (list[float]).

        Args:
            documents: List of document dictionaries.

        Returns:
            Number of documents indexed in this call.

        Raises:
            ValueError: If a document is missing ``doc_id`` or ``content``.
        """
        parsed: list[Document] = []
        for raw in documents:
            if "doc_id" not in raw or "content" not in raw:
                raise ValueError(
                    f"Each document must have 'doc_id' and 'content' keys; got: {list(raw.keys())}"
                )
            parsed.append(
                Document(
                    doc_id=str(raw["doc_id"]),
                    content=str(raw["content"]),
                    metadata=raw.get("metadata", {}),
                    embedding=raw.get("embedding"),
                )
            )

        self._engine.index_documents(parsed)
        self._doc_count += len(parsed)
        logger.info("RetrievalIndex: indexed %d new documents (total=%d).", len(parsed), self._doc_count)
        return len(parsed)

    def search(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Search the index using Hybrid BM25 + Dense RRF.

        Args:
            query: Natural language text query.
            query_embedding: Optional dense embedding vector for semantic ranking.
            top_k: Maximum number of results to return.

        Returns:
            Ordered list of ``SearchResult`` objects (highest RRF score first).

        Raises:
            ValueError: If the index is empty.
        """
        if self._doc_count == 0:
            raise ValueError(
                "RetrievalIndex is empty. Call index_documents() before searching."
            )
        return self._engine.search(query=query, query_embedding=query_embedding, top_k=top_k)

    def clear(self) -> None:
        """Remove all indexed documents and reset the engine state."""
        self._engine = HybridSearchEngine(rrf_k=self._engine.rrf_k)
        self._doc_count = 0
        logger.info("RetrievalIndex: cleared all indexed documents.")

    def to_tool_result(self, results: list[SearchResult]) -> dict[str, Any]:
        """Serialise ``SearchResult`` objects to a JSON-safe tool output dict.

        Args:
            results: List of ``SearchResult`` from ``search()``.

        Returns:
            Dict with ``hits`` list and ``total_hits`` count, suitable as
            ``ObservationPayload.output_data``.
        """
        return {
            "total_hits": len(results),
            "hits": [
                {
                    "doc_id": r.document.doc_id,
                    "content": r.document.content,
                    "metadata": r.document.metadata,
                    "rrf_score": r.rrf_score,
                    "bm25_rank": r.bm25_rank,
                    "dense_rank": r.dense_rank,
                }
                for r in results
            ],
        }


__all__ = ["RetrievalIndex"]
