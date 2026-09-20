# Model Merging: Architecture Assessment & Decision Record (ARS-T08)

## 1. Executive Summary & Decision

**Decision**: **ADOPT [`mergekit`](https://github.com/arcee-ai/mergekit) for all full-model merging workflows.** Do **not** build, expand, or maintain bespoke production merging code in this repository.

The repository's internal merging module ([`src/merging/`](file:///c:/Users/vinicius/Documents/GeminiCodes/Athena-reasoning-sandbox/src/merging/)) is retained strictly as an **educational and testing toolkit** for lightweight, in-memory synthetic tensor demonstrations (Phase 1d in `src/main.py`) and algorithmic unit tests. All production-scale neural model merges must use the upstream `mergekit` framework.

---

## 2. In-Repo Merging Toolkit ([`src/merging/`](file:///c:/Users/vinicius/Documents/GeminiCodes/Athena-reasoning-sandbox/src/merging/))

### Scope & Role
The in-repo merging package provides clean, pure-PyTorch implementations of three foundational tensor operations:
1. **SLERP** (*Spherical Linear Interpolation*): Geometric interpolation along the hypersphere surface maintaining constant angular velocity and norm geometry.
2. **TIES-Merging** (*Trim, Elect Sign & Merge*): Parameter interference resolution via top-$k$ magnitude trimming, majority sign election, and disjoint mean calculation.
3. **DARE** (*Drop And REscale*): Bernoulli delta pruning with compensation scaling ($1 / (1 - p)$).

### Hard Architectural Limitations
While suitable for synthetic tensor verification, `src/merging/` is fundamentally unsuited for merging real-world LLMs:
- **In-Memory Materialization**: Computes merges by loading all parameter tensors directly into memory simultaneously. Attempting to merge real models (e.g. 7B, 13B, 70B parameters) results in immediate Out-Of-Memory (OOM) errors.
- **No Safetensors / Disk Streaming**: Lacks out-of-core layer-wise streaming and shard-aware parameter extraction.
- **No Tokenizer Harmonization**: Does not reconcile vocabulary differences, token ID remapping, or embedding resizing across parent models.
- **No Cross-Architecture Support**: Assumes strictly identical parameter dimensions and layer naming conventions without normalization.
- **Limited Operator Set**: Does not support modern advanced fusion algorithms such as DELLA, DARE-TIES, Passthrough/Frankenmerging, Model Breadcrumbs, or Mixture-of-Experts (MoE) routing.

---

## 3. Upstream Mergekit Framework Assessment (S-ARS-6)

[`mergekit`](https://github.com/arcee-ai/mergekit) (developed by Charles Goddard and Arcee AI) is the open-source industry standard for combining pretrained and fine-tuned language models.

### Key Capabilities
- **Out-of-Core Streaming Execution**: Iterates through weights layer by layer, loading only the active layer tensors into RAM/VRAM and writing merged outputs directly to disk. Enables merging 70B+ models on commodity workstations or CPU/NVMe with as little as 16–32 GB RAM.
- **Exhaustive Fusion Methods**: Implements Linear, SLERP, TIES, DARE (DARE-TASK, DARE-TIES), Task Arithmetic, Passthrough (depth concatenation / Frankenmerging), DELLA, Model Breadcrumbs, and automated MoE router synthesis.
- **Robust Tokenizer Reconciliation**: Automatically resolves tokenizer disparities, merges vocabularies, and aligns embedding layers.
- **Format Flexibility**: Native support for sharded Safetensors, Hugging Face Hub downloads, PyTorch checkpoints, and direct GGUF export.
- **Precision & Quantization Handling**: Merges in float32, float16, or bfloat16, with support for quantized weight representations.

---

## 4. Build vs. Adopt Trade-Off Matrix

| Criterion | In-Repo Bespoke (`src/merging/`) | Upstream `mergekit` (Adopt) | Evaluation & Rationale |
| :--- | :--- | :--- | :--- |
| **Correctness & Robustness** | Basic mathematical implementation; untested on edge-case numerical geometries or tied weights. | Battle-tested across thousands of published models on the Hugging Face Open LLM Leaderboard. | **Advantage: mergekit**. Mitigates subtle numerical bugs (e.g., SLERP collinearity singularities when $\theta \to 0$, sign election edge cases). |
| **Memory Scalability** | $O(N \times \text{Params})$ RAM/VRAM footprint; fails on models $>1\text{B}$ parameters. | $O(\text{Single Layer})$ RAM footprint with out-of-core disk streaming. | **Advantage: mergekit**. Essential for real-world viability. |
| **Tokenizer Handling** | None (tensors only). | Automated tokenizer unification and embedding alignment. | **Advantage: mergekit**. Bespoke tokenizer merging is complex and error-prone. |
| **Maintenance Burden** | High: requires continuous updates as new architectures emerge (Llama 3, Mistral, Gemma, Qwen, DeepSeek). | Zero internal maintenance: updates are maintained upstream by active community. | **Advantage: mergekit**. Offloads architectural drift. |
| **Licensing** | MIT (this repository). | Apache 2.0. | **Compatible**: Apache 2.0 dependencies can be freely utilized alongside MIT repositories. |
| **Local Experimentation** | Instant, zero-install dependency for synthetic tensor unit tests. | Requires installing heavy external CLI/tooling. | **Advantage: In-Repo**. Justifies retaining `src/merging/` for tests and lightweight demos. |

---

## 5. Usage Guidelines

### When to Use In-Repo Code (`src/merging/`)
- Unit testing pure tensor fusion math (`tests/test_merging_and_reasoning.py`).
- Synthetic tensor demonstrations (`python -m src.main` Phase 1d).
- Prototyping new mathematical tensor operations before formal submission upstream.

### When to Use `mergekit`
- Merging any real model checkpoints from Hugging Face or local storage.
- Combining models with different fine-tuning targets (e.g., coding + math + chat).
- Constructing Frankenmerges (depth extension) or MoE architectures.

### Production Mergekit Workflow
The repository includes a ready-to-run Mergekit specification at [`configs/merge_config.yaml`](file:///c:/Users/vinicius/Documents/GeminiCodes/Athena-reasoning-sandbox/configs/merge_config.yaml).

To execute a full-model merge using `mergekit`:
```bash
# 1. Install mergekit
pip install mergekit

# 2. Run the YAML-driven merge
mergekit-yaml configs/merge_config.yaml ./output-merged-model --copy-tokenizer --cuda --low-cpu-memory

# 3. Verify the merged model
python -c "from transformers import AutoModelForCausalLM; model = AutoModelForCausalLM.from_pretrained('./output-merged-model')"
```
