# Hybrid RAG & Retrieval Engine Guide

This guide explains the architecture, algorithmic mechanics, and operational usage of the **Hybrid Search Engine** and stateful **Retrieval Index** in the Athena Reasoning Sandbox.

---

## 1. Algorithmic Foundation: Hybrid BM25 + Dense RRF

Traditional RAG systems frequently suffer from vocabulary mismatches when using dense embeddings alone, or semantic blindness when using keyword search alone.

Athena resolves this through **Reciprocal Rank Fusion (RRF)**:

1. **Sparse Lexical Ranking (BM25)**:
   Scores exact keyword matches using inverse document frequency (IDF) and document length normalization:
   $$\text{score}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{f(t, D) \cdot (k_1 + 1)}{f(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

2. **Dense Semantic Ranking (Cosine Similarity)**:
   Computes vector dot products across query and document dense embeddings.

3. **Reciprocal Rank Fusion (RRF)**:
   Merges both rank orderings into a single consolidated score:
   $$\text{RRF}(d) = \frac{1}{k + \text{rank}_{\text{BM25}}(d)} + \frac{1}{k + \text{rank}_{\text{Dense}}(d)}$$
   where $k = 60$ is the standard smoothing constant preventing outliers from dominating results.

---

## 2. Stateful Retrieval Index (`RetrievalIndex`)

The `RetrievalIndex` (`src/rag/retrieval_index.py`) provides an in-process, zero-dependency document store with:

- **Upsert Semantics**: Inserting documents with existing `doc_id` updates their content and rebuilds indices automatically.
- **Document Management**: Lookup (`get_document`), deletion (`delete_document`), and bulk export (`export_documents`).
- **Disk Persistence**: Serializing to JSON (`save_to_json`) and restoring from disk (`load_from_json`).

### Python Example: Indexing & Persistence

```python
from src.rag.retrieval_index import RetrievalIndex

# Initialize index
index = RetrievalIndex(rrf_k=60)

# Ingest documents
index.index_documents([
    {
        "doc_id": "guide-finops",
        "content": "FinOps principle: enforce partition pruning on analytical queries.",
        "metadata": {"category": "governance"},
    },
    {
        "doc_id": "guide-slerp",
        "content": "SLERP combines model weights on the unit hypersphere surface.",
        "metadata": {"category": "machine-learning"},
    },
])

# Save index state to disk
index.save_to_json("data/knowledge_base.json")

# Restore into a new index instance
new_index = RetrievalIndex()
new_index.load_from_json("data/knowledge_base.json")

# Hybrid Search
results = new_index.search(query="partition pruning for query cost", top_k=1)
for res in results:
    print(f"Top Hit: {res.document.doc_id} (RRF Score: {res.rrf_score:.4f})")
```

---

## 3. Tool Dispatch Across Parallax & MCP

When dispatched via `ExecutiveEngineProcess` or `AthenaMCPServer`, the `retrieval_search` tool accepts:

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `op` / `operation` | string | No | Operation: `"search"`, `"index"`, `"clear"`, `"export"`, `"get"`, `"delete"`, `"save"`, `"load"`. Default is `"search"`. |
| `query` | string | For search | Search query string. |
| `top_k` | int | No | Maximum hits to return (default 5). |
| `filter_metadata` | dict | No | Key-value dictionary to pre-filter candidate documents. |
| `documents` | list[dict] | For index | Document dictionaries with `doc_id` and `content`. |
| `doc_id` | string | For get/delete | Document ID to inspect or remove. |
| `filepath` | string | For save/load | Path to destination or source JSON file. |

---

---

## 4. Deterministic Offline RAG Evaluation (Lane 1)

Athena provides zero-dependency, deterministic RAG evaluation heuristics (`evals/metrics/rag_metrics.py`) that run 100% offline in CI without paid LLM API keys:

1. **Context Relevancy (`DeterministicContextRelevancyMetric`)**:
   - Assesses what proportion of retrieved chunks contain query keywords or key semantic terms.
2. **Context Precision (`DeterministicContextPrecisionMetric`)**:
   - Evaluates rank-weighted positioning via Mean Average Precision (MAP / Precision@k), penalizing retrievals where irrelevant contexts are placed before relevant ones.
3. **Output Faithfulness (`DeterministicFaithfulnessMetric`)**:
   - Evaluates whether factual claims in the generated output are grounded in the retrieved contexts rather than hallucinated.
4. **Unified Composite Gate (`DeterministicRAGGate`)**:
   - Combines Relevancy (35%), Precision (35%), and Faithfulness (30%) with a configurable pass threshold (default 0.70).

```python
from deepeval.test_case import LLMTestCase
from evals.metrics.rag_metrics import DeterministicRAGGate

test_case = LLMTestCase(
    input="What is AWS Athena pricing model?",
    actual_output="AWS Athena charges 5 dollars per TB scanned.",
    retrieval_context=["AWS Athena charges $5.00 per terabyte of data scanned from Amazon S3."],
)

gate = DeterministicRAGGate(threshold=0.70)
score = gate.measure(test_case)
assert gate.is_successful()
```

---

## 5. Verification & Testing

Run the dedicated test suites to validate hybrid search, persistence, and deterministic RAG evaluation:

```bash
# Run hybrid search tests:
pytest tests/test_retrieval_search.py -v

# Run persistence and CRUD tests:
pytest tests/test_retrieval_persistence.py -v

# Run deterministic RAG evaluation tests:
pytest evals/test_rag_evaluation.py -v
```

