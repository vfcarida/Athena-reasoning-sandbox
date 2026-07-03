<div align="center">
  <h1>🧠 Athena Reasoning Sandbox</h1>
  <p><b>Scientific Framework for Neural Model Experimentation</b></p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python Version">
    <img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c" alt="PyTorch">
    <img src="https://img.shields.io/badge/Transformers-4.35%2B-yellow" alt="Hugging Face">
    <img src="https://img.shields.io/badge/Status-Experimental-orange" alt="Status">
  </p>
</div>

---

> [!NOTE]
> This repository implements the core **model development and training** pipelines for our corporate AI strategy. All model evaluation and grading is delegated to the companion [Agent-Bench](https://github.com/vfcarida/Agent-Bench) repository.

## 🌟 Strategic Overview

Athena Reasoning Sandbox is a research-grade Python framework designed for the full lifecycle of language model specialization. It addresses the critical "Build vs. Buy" corporate dilemma by providing highly efficient paths to create sovereign, domain-specific AI models that keep proprietary data strictly internal.

We implement the four essential pathways of model specialization:

| Phase | Methodology | Target Use Case | Module |
|-------|-------------|-----------------|---------|
| **1a** | **From-Scratch Pretraining** | Extreme data sovereignty & novel vocabularies | `src/pretraining/from_scratch.py` |
| **1b** | **Continued Pretraining** | Deep domain adaptation of open weights | `src/pretraining/continued_pretraining.py` |
| **1c** | **SFT & PEFT (LoRA/QLoRA)** | Instruction following & style alignment | `src/finetuning/` |
| **1d** | **Merging & SwiReasoning** | Multi-skill fusion & entropy-guided inference | `src/merging/`, `src/reasoning/` |

---

## 🏗️ Architecture & Data Flow

```mermaid
graph TD
    A[Raw Corporate Data] -->|Pretraining| B(Phase 1a & 1b)
    B -->|Base Models| C{Athena Sandbox}
    
    C -->|Adapter Injection| D[Phase 1c: LoRA/QLoRA]
    C -->|Tensor Fusion| E[Phase 1d: Merging]
    
    D --> F[Specialized Model]
    E --> F
    
    F -->|SwiReasoning| G[Adaptive Inference Engine]
    F -.->|Export via Bridge| H((Agent-Bench))
    
    style C fill:#2b3137,stroke:#fff,stroke-width:2px,color:#fff
    style H fill:#d32f2f,stroke:#fff,stroke-width:2px,color:#fff
```

### Directory Structure
```text
Athena-reasoning-sandbox/
├── configs/                   # YAML configs for all training pipelines
├── src/
│   ├── main.py                # 🚀 Orchestrator (Demo all pathways)
│   ├── pretraining/           # 🧬 From-scratch & continued pretraining (AMP, RoPE)
│   ├── finetuning/            # 🛠️ SFT, LoRA adapters, 4-bit quantization
│   ├── merging/               # 🔗 SLERP, TIES, DARE tensor operations
│   ├── reasoning/             # 🤔 SwiReasoning entropy simulator
│   ├── bridge/                # 🌉 Agent-Bench evaluation connector
│   └── utils/                 # 📊 Mathematical metrics (Shannon entropy, Elo)
└── tests/                     # 🧪 Unit test suite
```

---

## 🚀 Installation & Quick Start

1. **Clone & Environment Setup**
```bash
git clone https://github.com/vfcarida/Athena-reasoning-sandbox.git
cd Athena-reasoning-sandbox

python -m venv .venv
# Activate: `.venv\Scripts\activate` (Win) or `source .venv/bin/activate` (Mac/Linux)
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt

# [Optional] For QLoRA 4-bit fine-tuning (Requires NVIDIA GPU)
pip install bitsandbytes>=0.41.0

# [Optional] For full evaluation integration
pip install -e ../Agent-Bench
```

3. **Run the Sandbox Demo**
Execute the main orchestrator to see all 4 pathways simulated:
```bash
python -m src.main
```

---

## 🔬 Mathematical Foundations

> [!TIP]
> **Why we use these algorithms:**
> Understanding the math allows us to push boundaries beyond basic API calls.

### 1. LoRA (Low-Rank Adaptation)
Instead of updating the massive weight matrix $W$, LoRA freezes $W$ and trains a low-rank decomposition $A$ and $B$:
$$W' = W + \frac{\alpha}{r} (B \cdot A)$$
Where $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times d}$, with $r \ll d$.

### 2. SLERP (Spherical Linear Interpolation)
Unlike simple averaging, SLERP preserves the angular geometric properties of parameter vectors during model merging:
$$\text{Slerp}(p_0, p_1; t) = \frac{\sin((1-t)\theta)}{\sin\theta} p_0 + \frac{\sin(t\theta)}{\sin\theta} p_1$$

### 3. SwiReasoning (Entropy-Guided)
The model calculates Shannon Entropy for its current prediction probability distribution $P(x)$:
$$H(X) = -\sum_{i=1}^{M} P(x_i) \log_2 P(x_i)$$
If $H(X) > \text{threshold}$, the model injects a `<think>` token to expand latent reasoning capacity before emitting the final answer.

---

## 🧪 RAG TDD Evaluation Pathway (DeepEval & SageMaker)

To ensure high-quality information generation and retrieval, Athena incorporates a rigid **CI/CD Quality Gate** evaluating the **RAG Triad**:

1. **Faithfulness (Groundedness)**: Assesses if the actual output is factually aligned with the retrieval context.
2. **Answer Relevance**: Measures how well the output addresses the user's query.
3. **Context Precision**: Evaluates the ranking quality of the retriever (i.e. whether relevant nodes are placed at the top).

The pipeline uses the **LLM-as-a-Judge** pattern, querying an external **AWS SageMaker Endpoint** as the evaluator.

### 🏗️ RAG Evaluation Quality Gate Architecture

```mermaid
graph TD
    subgraph RAG Pipeline
        A[User Query] --> B[Retriever DB]
        B -->|Context Documents| C[Generator LLM]
        A --> C
        C -->|Generated Answer| D[RAG Test Case]
    end
    
    subgraph CI/CD Quality Gate
        D -->|Evaluate Triad| E{deepeval Evaluator}
        E -->|1. Faithfulness Metric| F{Score >= 0.85?}
        E -->|2. Answer Relevance| G{Score >= 0.85?}
        E -->|3. Context Precision| H{Score >= 0.85?}
        
        F -->|Pass| I[Collect Score]
        G -->|Pass| I
        H -->|Pass| I
        
        F -->|Fail| J[Raise AssertionError + Log Reason]
        G -->|Fail| J
        H -->|Fail| J
        
        I -->|All Passed| K[CI/CD Build Green]
        J -->|Any Failed| L[CI/CD Build Red / Blocked]
    end
    
    subgraph LLM Judge
        E -.->|Query Judge| M[AWS SageMaker Runtime]
        M -.->|Mocked in Tests| N[MockSageMakerLLM]
    end
    
    style E fill:#2b3137,stroke:#fff,stroke-width:2px,color:#fff
    style K fill:#2e7d32,stroke:#fff,stroke-width:2px,color:#fff
    style L fill:#c62828,stroke:#fff,stroke-width:2px,color:#fff
```

### 🚀 Usage Guide

```python
from src.utils.rag_evaluator import SageMakerLLM, evaluate_rag_triad

# 1. Instantiate SageMaker Judge Model
judge = SageMakerLLM(
    endpoint_name="my-sagemaker-llama3-endpoint",
    region_name="us-east-1",
    model_name="Llama 3 Judge"
)

# 2. Run RAG evaluation against the strict threshold (default >= 0.85)
results = evaluate_rag_triad(
    input_text="What is our return policy?",
    actual_output="You can return any product within 30 days of purchase.",
    retrieval_context=["Our return policy allows items to be returned within 30 days."],
    expected_output="Items can be returned within 30 days.",
    threshold=0.85,
    judge_model=judge
)
```

### 🔬 Testing and CI/CD Verification

Our test suite uses a dynamic `MockSageMakerLLM` to safely simulate structured Pydantic verdicts without invoking AWS or generating computational costs.

Run the RAG TDD test suite locally:
```bash
python -m pytest tests/test_rag_evaluator.py -v
```

---

## 🌉 Agent-Bench Integration

Models built in Athena are not graded here. They are exported via the `AgentBenchBridge` to be rigorously evaluated against semantic and functional benchmarks.


```python
from src.bridge.agent_bench_bridge import AgentBenchBridge

# 1. Initialize the bridge
bridge = AgentBenchBridge()

# 2. Export your newly trained model
bridge.export_for_evaluation(model, tokenizer, "./models/finance-specialist")

# 3. Trigger external evaluation suite
results = bridge.run_benchmark(suite_id="finance_reasoning_v1")
```

---
<div align="center">
  <i>Developed for Advanced Agentic Coding & MLOps Research</i>
</div>
