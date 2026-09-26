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

import json
import logging
from pathlib import Path
from typing import Any

from src.rag.hybrid_search import Document, HybridSearchEngine, SearchResult

logger = logging.getLogger(__name__)


class RetrievalIndex:
    """Stateful in-process document retrieval index backed by ``HybridSearchEngine``.

    Designed to be injected into ``ExecutiveEngineProcess`` so that the
    ``retrieval_search`` tool handler has access to a live, query-able document store.
    Supports upsert semantics, document export, and JSON file persistence.

    Attributes:
        _engine: Underlying ``HybridSearchEngine`` instance.
        _documents: Internal dictionary mapping document IDs to Document instances.
        _doc_count: Running total of documents indexed.
    """

    def __init__(self, rrf_k: int = 60) -> None:
        """Initialise the retrieval index.

        Args:
            rrf_k: Reciprocal Rank Fusion smoothing constant (default 60).
        """
        self._engine = HybridSearchEngine(rrf_k=rrf_k)
        self._documents: dict[str, Document] = {}
        self._doc_count: int = 0

    @property
    def document_count(self) -> int:
        """Return total number of unique indexed documents."""
        return len(self._documents)

    def index_documents(
        self,
        documents: list[dict[str, Any]],
        upsert: bool = True,
    ) -> int:
        """Add or update documents in the index.

        Each document dict must contain ``doc_id`` and ``content`` keys.
        Optional keys: ``metadata`` (dict) and ``embedding`` (list[float]).

        Args:
            documents: List of document dictionaries.
            upsert: If True (default), updates existing documents with matching doc_id.
                If False and doc_id already exists, skips the duplicate document.

        Returns:
            Number of documents added or updated in this call.

        Raises:
            ValueError: If a document is missing ``doc_id`` or ``content``.
        """
        modified_count = 0
        for raw in documents:
            if "doc_id" not in raw or "content" not in raw:
                raise ValueError(
                    f"Each document must have 'doc_id' and 'content' keys; got: {list(raw.keys())}"
                )
            doc_id = str(raw["doc_id"])
            if not upsert and doc_id in self._documents:
                continue

            doc = Document(
                doc_id=doc_id,
                content=str(raw["content"]),
                metadata=raw.get("metadata", {}),
                embedding=raw.get("embedding"),
            )
            self._documents[doc_id] = doc
            modified_count += 1

        # Rebuild underlying search engine with current document set
        self._engine = HybridSearchEngine(rrf_k=self._engine.rrf_k)
        self._engine.index_documents(list(self._documents.values()))
        self._doc_count = len(self._documents)
        logger.info(
            "RetrievalIndex: indexed %d documents (total=%d unique).",
            modified_count,
            self._doc_count,
        )
        return modified_count

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """Retrieve an indexed document dictionary by its unique doc_id.

        Args:
            doc_id: Unique document identifier.

        Returns:
            Dictionary with document fields, or None if not found.
        """
        doc = self._documents.get(str(doc_id))
        if doc is None:
            return None
        return {
            "doc_id": doc.doc_id,
            "content": doc.content,
            "metadata": doc.metadata,
            "embedding": doc.embedding,
        }

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by its doc_id from the index.

        Args:
            doc_id: Identifier of the document to delete.

        Returns:
            True if document existed and was deleted, False otherwise.
        """
        key = str(doc_id)
        if key not in self._documents:
            return False

        del self._documents[key]
        self._engine = HybridSearchEngine(rrf_k=self._engine.rrf_k)
        self._engine.index_documents(list(self._documents.values()))
        self._doc_count = len(self._documents)
        logger.info("RetrievalIndex: deleted document '%s' (remaining=%d).", key, self._doc_count)
        return True

    def export_documents(self) -> list[dict[str, Any]]:
        """Export all indexed documents as a list of dictionaries.

        Returns:
            List of document dictionaries suitable for JSON serialisation.
        """
        return [
            {
                "doc_id": doc.doc_id,
                "content": doc.content,
                "metadata": doc.metadata,
                "embedding": doc.embedding,
            }
            for doc in self._documents.values()
        ]

    def save_to_json(self, filepath: str | Path) -> None:
        """Persist the index documents to a JSON file on disk.

        Args:
            filepath: Destination file path.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        docs = self.export_documents()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(docs, f, indent=2, ensure_ascii=False)
        logger.info("RetrievalIndex: saved %d documents to '%s'.", len(docs), path)

    def load_from_json(self, filepath: str | Path, clear_existing: bool = False) -> int:
        """Load documents from a JSON file on disk into the index.

        Args:
            filepath: Source file path.
            clear_existing: If True, clears existing index before loading.

        Returns:
            Number of documents indexed from the file.

        Raises:
            FileNotFoundError: If filepath does not exist.
            ValueError: If file content is not a valid list of document dicts.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Index file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list of documents, got {type(data).__name__}")

        if clear_existing:
            self.clear()

        return self.index_documents(data, upsert=True)

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
        self._documents.clear()
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
