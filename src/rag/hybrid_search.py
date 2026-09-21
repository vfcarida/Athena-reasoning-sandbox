"""Hybrid Search Engine combining Lexical BM25 and Dense Semantic Search.

Combines sparse keyword matching (BM25) with dense semantic embeddings (e.g. text-embedding-ada-002)
using Reciprocal Rank Fusion (RRF). RRF re-ranks results to eliminate the "Lost in the Middle"
contextual degradation effect in long-horizon LLM reasoning prompts.

Algorithmic Complexity:
    - BM25 Scoring: O(N * L) where N is document count and L is average doc token length.
    - Dense Cosine Similarity: O(N * D) where D is embedding vector dimension.
    - Reciprocal Rank Fusion (RRF): O(N log N) sorting step per fusion.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document chunk in the search index.

    Attributes:
        doc_id: Unique string document identifier.
        content: Text content of the document chunk.
        metadata: Key-value dictionary metadata.
        embedding: Dense float embedding vector representation.
    """
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class SearchResult:
    """SearchResult container with hybrid relevance scores.

    Attributes:
        document: Target retrieved Document object.
        rrf_score: Reciprocal Rank Fusion unified score.
        bm25_rank: Rank in lexical BM25 matching.
        dense_rank: Rank in dense semantic search.
    """
    document: Document
    rrf_score: float
    bm25_rank: int
    dense_rank: int


class HybridSearchEngine:
    """Hybrid search engine with Reciprocal Rank Fusion (RRF).

    Attributes:
        documents: Index of Document objects.
        rrf_k: Smoothing constant for RRF formula (default 60).
    """

    def __init__(self, rrf_k: int = 60) -> None:
        """Initialize Hybrid Search Engine.

        Args:
            rrf_k: RRF smoothing constant k. Default 60.
        """
        self.documents: List[Document] = []
        self.rrf_k = rrf_k

    def index_documents(self, docs: List[Document]) -> None:
        """Add document chunks to the hybrid index.

        Args:
            docs: List of Document objects.
        """
        self.documents.extend(docs)
        logger.info("Indexed %d documents into HybridSearchEngine.", len(docs))

    def _compute_bm25_scores(self, query: str) -> List[Tuple[Document, float]]:
        """Compute BM25 sparse keyword matching scores.

        Args:
            query: User search query string.

        Returns:
            List of (Document, score) tuples sorted by score descending.
        """
        query_terms = set(query.lower().split())
        results: List[Tuple[Document, float]] = []

        avg_dl = sum(len(doc.content.split()) for doc in self.documents) / max(len(self.documents), 1)
        k1 = 1.5
        b = 0.75

        for doc in self.documents:
            doc_terms = doc.content.lower().split()
            dl = len(doc_terms)
            score = 0.0

            for term in query_terms:
                freq = doc_terms.count(term)
                if freq > 0:
                    idf = math.log((len(self.documents) + 1) / (1 + sum(1 for d in self.documents if term in d.content.lower())))
                    num = freq * (k1 + 1)
                    den = freq + k1 * (1 - b + b * (dl / max(avg_dl, 1.0)))
                    score += idf * (num / max(den, 1e-6))

            results.append((doc, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _compute_dense_scores(self, query_embedding: List[float]) -> List[Tuple[Document, float]]:
        """Compute dense cosine similarity scores against document embeddings.

        Args:
            query_embedding: Dense float vector for search query.

        Returns:
            List of (Document, cosine_similarity) tuples sorted by score descending.
        """
        results: List[Tuple[Document, float]] = []

        def cosine_sim(a: List[float], b: List[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(y * y for y in b))
            return dot / max(norm_a * norm_b, 1e-9)

        for doc in self.documents:
            if doc.embedding is not None and len(doc.embedding) == len(query_embedding):
                sim = cosine_sim(query_embedding, doc.embedding)
            else:
                sim = 0.0
            results.append((doc, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def search(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Perform hybrid BM25 + Dense Semantic search combined via Reciprocal Rank Fusion (RRF).

        Formula:
            RRF_score(d) = sum_{m in M} (1 / (k + rank_m(d)))

        Args:
            query: Text query string.
            query_embedding: Optional dense embedding for semantic matching.
            top_k: Maximum number of results to return.

        Returns:
            Ordered list of SearchResult objects.
        """
        bm25_ranked = self._compute_bm25_scores(query)
        bm25_ranks = {doc.doc_id: i + 1 for i, (doc, _) in enumerate(bm25_ranked)}

        if query_embedding:
            dense_ranked = self._compute_dense_scores(query_embedding)
            dense_ranks = {doc.doc_id: i + 1 for i, (doc, _) in enumerate(dense_ranked)}
        else:
            dense_ranks = {doc.doc_id: len(self.documents) for doc in self.documents}

        rrf_scores: Dict[str, float] = {}
        for doc in self.documents:
            r_bm25 = bm25_ranks.get(doc.doc_id, len(self.documents))
            r_dense = dense_ranks.get(doc.doc_id, len(self.documents))

            score = (1.0 / (self.rrf_k + r_bm25)) + (1.0 / (self.rrf_k + r_dense))
            rrf_scores[doc.doc_id] = score

        sorted_docs = sorted(self.documents, key=lambda d: rrf_scores[d.doc_id], reverse=True)

        output: List[SearchResult] = []
        for doc in sorted_docs[:top_k]:
            output.append(
                SearchResult(
                    document=doc,
                    rrf_score=round(rrf_scores[doc.doc_id], 6),
                    bm25_rank=bm25_ranks.get(doc.doc_id, 0),
                    dense_rank=dense_ranks.get(doc.doc_id, 0),
                )
            )

        return output

    @staticmethod
    def evaluate_faithfulness(
        query: str,
        actual_output: str,
        retrieved_contexts: List[str],
        threshold: float = 0.75,
    ) -> float:
        """Evaluate RAG faithfulness using DeepEval's FaithfulnessMetric.

        Args:
            query: Input prompt.
            actual_output: Generated output text.
            retrieved_contexts: List of retrieved context strings.
            threshold: Passing score threshold.

        Returns:
            Faithfulness score between 0.0 and 1.0.
        """
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            retrieval_context=retrieved_contexts,
        )
        metric = FaithfulnessMetric(threshold=threshold)
        score = metric.measure(test_case)
        return score
