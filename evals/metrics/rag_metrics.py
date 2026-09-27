"""Deterministic Evaluation Metrics for Retrieval-Augmented Generation (RAG).

Provides honest, deterministic, Lane-1 offline metrics for evaluating RAG pipelines:
1. DeterministicContextRelevancyMetric: Evaluates whether retrieved contexts match query intent.
2. DeterministicContextPrecisionMetric: Evaluates rank-weighted ordering (Precision@k / MAP).
3. DeterministicFaithfulnessMetric: Evaluates groundedness of output claims against retrieved context.
4. DeterministicRAGGate: Composite gate combining relevancy, precision, and faithfulness.

HONESTY NOTICE:
    These metrics are DETERMINISTIC HEURISTICS and rule-based evaluators executing entirely
    offline in Lane 1 without external LLM API calls or paid judge dependencies.
"""

from __future__ import annotations

import re
from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

# Standard analytical and query stopwords
_STOPWORDS: set[str] = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with",
    "by", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "from", "up", "down", "of", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now", "what", "which", "who",
    "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing",
}


def extract_text(item: Any) -> str:
    """Extract plain text from string or context data object.

    Args:
        item: Raw text string or object with content attribute.

    Returns:
        Extracted string content.
    """
    if isinstance(item, str):
        return item
    if hasattr(item, "content"):
        return str(item.content)
    return str(item)


def extract_keywords(text: Any) -> set[str]:
    """Extract lowercase significant keywords (length >= 2, non-stopword).

    Args:
        text: Input string or context object.

    Returns:
        Set of lowercase filtered keywords.
    """
    raw_str = extract_text(text)
    tokens = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", raw_str.lower())
    return {t for t in tokens if t not in _STOPWORDS}


def extract_claims(text: str) -> list[str]:
    """Extract substantive sentences or clauses representing verifiable claims.

    Args:
        text: Generated output text.

    Returns:
        List of trimmed non-empty claim sentences.
    """
    sentences = re.split(r"(?<=[.!?\n])\s+", text.strip())
    claims = []
    for s in sentences:
        cleaned = s.strip()
        if len(cleaned) >= 15 and not cleaned.startswith("<think>"):
            claims.append(cleaned)
    return claims or ([text.strip()] if text.strip() else [])


class DeterministicContextRelevancyMetric(BaseMetric):
    """Deterministic heuristic evaluating whether retrieved contexts are relevant to the query.

    Computes the proportion of retrieved chunks that contain query keywords.
    Score = (Relevant Chunks) / (Total Retrieved Chunks).
    """

    def __init__(self, threshold: float = 0.70, min_keyword_overlap: float = 0.20) -> None:
        super().__init__()
        self.threshold = threshold
        self.min_keyword_overlap = min_keyword_overlap
        self.score: float = 0.0
        self.reason: str = ""
        self.success: bool = False
        self.details: dict[str, Any] = {}

    @property
    def __name__(self) -> str:
        return "DeterministicContextRelevancyMetric"

    def measure(self, test_case: LLMTestCase) -> float:
        query = (test_case.input or "").strip()
        contexts = test_case.retrieval_context or []

        if not contexts:
            self.score = 0.0
            self.reason = "No retrieval context provided. Relevancy score: 0.0."
            self.success = False
            return self.score

        query_kw = extract_keywords(query)
        if not query_kw:
            self.score = 1.0
            self.reason = "Query contains only generic stopwords; defaulting relevancy to 1.0."
            self.success = True
            return self.score

        relevant_count = 0
        per_context_overlap: list[float] = []

        for ctx in contexts:
            ctx_kw = extract_keywords(ctx)
            intersect = query_kw.intersection(ctx_kw)
            overlap = len(intersect) / len(query_kw) if query_kw else 0.0
            per_context_overlap.append(round(overlap, 4))
            if overlap >= self.min_keyword_overlap or len(intersect) >= 2:
                relevant_count += 1

        self.score = round(relevant_count / len(contexts), 4)
        threshold_val = float(self.threshold) if self.threshold is not None else 0.70
        self.success = self.score >= threshold_val
        self.details = {
            "query_keywords": sorted(query_kw),
            "relevant_count": relevant_count,
            "total_contexts": len(contexts),
            "overlaps": per_context_overlap,
        }
        self.reason = (
            f"Context Relevancy: {relevant_count}/{len(contexts)} contexts matched "
            f"(Score={self.score:.4f}, Threshold={threshold_val:.2f}, Pass={self.success})."
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        threshold_val = float(self.threshold) if self.threshold is not None else 0.70
        return bool(self.score is not None and self.score >= threshold_val)


class DeterministicContextPrecisionMetric(BaseMetric):
    """Deterministic heuristic evaluating rank-weighted retrieval precision.

    Evaluates whether relevant contexts are ranked near the top of the retrieval context list
    using Mean Average Precision (MAP) / Context Precision at k:
        Precision@k = (Relevant chunks up to position k) / k
        Context Precision = sum(Precision@k * is_relevant(k)) / total_relevant_chunks
    """

    def __init__(self, threshold: float = 0.70, min_keyword_overlap: float = 0.20) -> None:
        super().__init__()
        self.threshold = threshold
        self.min_keyword_overlap = min_keyword_overlap
        self.score: float = 0.0
        self.reason: str = ""
        self.success: bool = False
        self.details: dict[str, Any] = {}

    @property
    def __name__(self) -> str:
        return "DeterministicContextPrecisionMetric"

    def measure(self, test_case: LLMTestCase) -> float:
        query = (test_case.input or "").strip()
        contexts = test_case.retrieval_context or []

        if not contexts:
            self.score = 0.0
            self.reason = "No retrieval context provided. Precision score: 0.0."
            self.success = False
            return self.score

        query_kw = extract_keywords(query)
        if not query_kw:
            self.score = 1.0
            self.reason = "Query contains only generic stopwords; defaulting precision to 1.0."
            self.success = True
            return self.score

        relevant_flags: list[bool] = []
        for ctx in contexts:
            ctx_kw = extract_keywords(ctx)
            intersect = query_kw.intersection(ctx_kw)
            overlap = len(intersect) / len(query_kw) if query_kw else 0.0
            relevant_flags.append(overlap >= self.min_keyword_overlap or len(intersect) >= 2)

        total_relevant = sum(relevant_flags)
        if total_relevant == 0:
            self.score = 0.0
            self.reason = "Zero relevant contexts found in retrieval results. Precision: 0.0."
            self.success = False
            return self.score

        precisions_at_k: list[float] = []
        running_rel = 0
        for k, is_rel in enumerate(relevant_flags, start=1):
            if is_rel:
                running_rel += 1
                precisions_at_k.append(running_rel / k)

        self.score = round(sum(precisions_at_k) / total_relevant, 4)
        threshold_val = float(self.threshold) if self.threshold is not None else 0.70
        self.success = self.score >= threshold_val
        self.details = {
            "total_contexts": len(contexts),
            "relevant_flags": relevant_flags,
            "precisions_at_k": [round(p, 4) for p in precisions_at_k],
        }
        self.reason = (
            f"Context Precision: MAP={self.score:.4f} across {total_relevant} relevant chunks "
            f"(Threshold={threshold_val:.2f}, Pass={self.success})."
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        threshold_val = float(self.threshold) if self.threshold is not None else 0.70
        return bool(self.score is not None and self.score >= threshold_val)


class DeterministicFaithfulnessMetric(BaseMetric):
    """Deterministic heuristic evaluating whether generated claims are grounded in retrieved context.

    Verifies that claims in actual_output derive from the retrieval_context rather than hallucinating
    unsupported entities, numbers, or key terms.
    Score = (Grounded Claims) / (Total Claims).
    """

    def __init__(self, threshold: float = 0.70, min_claim_grounding: float = 0.40) -> None:
        super().__init__()
        self.threshold = threshold
        self.min_claim_grounding = min_claim_grounding
        self.score: float = 0.0
        self.reason: str = ""
        self.success: bool = False
        self.details: dict[str, Any] = {}

    @property
    def __name__(self) -> str:
        return "DeterministicFaithfulnessMetric"

    def measure(self, test_case: LLMTestCase) -> float:
        output_text = (test_case.actual_output or "").strip()
        contexts = test_case.retrieval_context or []

        if not output_text:
            self.score = 0.0
            self.reason = "Actual output is empty. Faithfulness score: 0.0."
            self.success = False
            return self.score

        if not contexts:
            self.score = 0.0
            self.reason = "No context provided to ground the output. Faithfulness score: 0.0."
            self.success = False
            return self.score

        all_context_kw = set()
        for ctx in contexts:
            all_context_kw.update(extract_keywords(ctx))

        claims = extract_claims(output_text)
        if not claims:
            self.score = 1.0
            self.reason = "No factual claims identified; defaulting faithfulness to 1.0."
            self.success = True
            return self.score

        grounded_claims = 0
        per_claim_grounding: list[float] = []

        for claim in claims:
            claim_kw = extract_keywords(claim)
            if not claim_kw:
                grounded_claims += 1
                per_claim_grounding.append(1.0)
                continue
            grounded_tokens = claim_kw.intersection(all_context_kw)
            grounding_ratio = len(grounded_tokens) / len(claim_kw)
            per_claim_grounding.append(round(grounding_ratio, 4))
            if grounding_ratio >= self.min_claim_grounding:
                grounded_claims += 1

        self.score = round(grounded_claims / len(claims), 4)
        threshold_val = float(self.threshold) if self.threshold is not None else 0.70
        self.success = self.score >= threshold_val
        self.details = {
            "total_claims": len(claims),
            "grounded_claims": grounded_claims,
            "claim_grounding_ratios": per_claim_grounding,
        }
        self.reason = (
            f"Faithfulness: {grounded_claims}/{len(claims)} claims grounded in retrieved context "
            f"(Score={self.score:.4f}, Threshold={threshold_val:.2f}, Pass={self.success})."
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        threshold_val = float(self.threshold) if self.threshold is not None else 0.70
        return bool(self.score is not None and self.score >= threshold_val)


class DeterministicRAGGate(BaseMetric):
    """Unified composite gate for deterministic RAG evaluation in Lane 1 CI.

    Combines:
    - Context Relevancy: 35% weight
    - Context Precision: 35% weight
    - Output Faithfulness: 30% weight
    """

    def __init__(self, threshold: float = 0.70) -> None:
        super().__init__()
        self.threshold = threshold
        self.score: float = 0.0
        self.reason: str = ""
        self.success: bool = False
        self.details: dict[str, float] = {}

    @property
    def __name__(self) -> str:
        return "DeterministicRAGGate"

    def measure(self, test_case: LLMTestCase) -> float:
        threshold_val = float(self.threshold) if self.threshold is not None else 0.70
        rel_metric = DeterministicContextRelevancyMetric(threshold=threshold_val)
        prec_metric = DeterministicContextPrecisionMetric(threshold=threshold_val)
        faith_metric = DeterministicFaithfulnessMetric(threshold=threshold_val)

        rel_score = rel_metric.measure(test_case)
        prec_score = prec_metric.measure(test_case)
        faith_score = faith_metric.measure(test_case)

        composite = (0.35 * rel_score) + (0.35 * prec_score) + (0.30 * faith_score)
        self.score = round(composite, 4)
        self.success = self.score >= threshold_val
        self.details = {
            "relevancy": rel_score,
            "precision": prec_score,
            "faithfulness": faith_score,
        }
        self.reason = (
            f"Deterministic RAG Gate: Relevancy={rel_score:.2f}, "
            f"Precision={prec_score:.2f}, Faithfulness={faith_score:.2f}. "
            f"Composite={self.score:.4f} (Threshold={threshold_val:.2f}, Pass={self.success})."
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        threshold_val = float(self.threshold) if self.threshold is not None else 0.70
        return bool(self.score is not None and self.score >= threshold_val)
