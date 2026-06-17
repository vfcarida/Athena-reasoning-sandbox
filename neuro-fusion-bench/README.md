# NeuroFusionBench

> **Scientific Framework for Neural Model Experimentation**
> Model Merging • Adaptive Inference • Agent Evaluation

---

## Overview

NeuroFusionBench is a production-grade Python framework for three pillars of AI model research:

1. **Model Merging** — Mathematical fusion of neural weight tensors using SLERP, TIES-Merging, and DARE algorithms.
2. **SwiReasoning** — Entropy-guided adaptive inference that dynamically switches between explicit generation and latent thinking modes.
3. **Agent-Bench** — Rigorous evaluation suite for AI agent behavior with deterministic metrics and LLM-as-a-Judge semantic scoring.

## Architecture

```
neuro-fusion-bench/
├── configs/
│   ├── merge_config.yaml          # Mergekit-compatible fusion configuration
│   └── eval_config.yaml           # Agent-Bench scenarios and judge settings
├── src/
│   ├── main.py                    # Orchestrator — runs all 3 demo pipelines
│   ├── merging/
│   │   ├── merge_operators.py     # SLERP, TIES, DARE tensor operations
│   │   └── merge_pipeline.py      # YAML-driven merge orchestrator
│   ├── reasoning/
│   │   └── swi_reasoning.py       # Entropy-guided inference engine
│   ├── evaluation/
│   │   ├── agent_bench.py         # Deterministic tool & plan metrics
│   │   └── judge.py               # LLM-as-a-Judge semantic evaluation
│   └── utils/
│       └── metrics.py             # Shannon entropy, Elo rating, OT index
└── requirements.txt
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/neuro-fusion-bench.git
cd neuro-fusion-bench

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

**Requirements**: Python 3.10+, PyTorch 2.0+, Hugging Face Transformers 4.35+

## Quick Start

Run the full demonstration pipeline:

```bash
python -m src.main
```

This executes three integrated demos:

### Phase 1: Model Merging
Synthetic tensor fusion using all three algorithms with cosine similarity and norm analysis.

### Phase 2: SwiReasoning
Simulated entropy-guided generation showing real-time mode switching between explicit and latent thinking.

### Phase 3: Agent-Bench
Evaluation of synthetic agent traces with Tool Correctness, Efficiency, Plan Adherence, and Elo ratings.

## Core Components

### TensorMergeOperators

```python
from src.merging import TensorMergeOperators

# SLERP — Spherical Linear Interpolation
merged = TensorMergeOperators.slerp(weights_a, weights_b, t=0.5)

# TIES — Trim, Elect Sign & Merge
merged = TensorMergeOperators.ties_merge(base, [model_a, model_b], threshold=0.2)

# DARE — Drop And REscale
merged = TensorMergeOperators.dare_drop_and_rescale(base, finetuned, drop_rate=0.3)
```

### SwiReasoningEngine

```python
from src.reasoning import SwiReasoningEngine

engine = SwiReasoningEngine(
    model=hf_model,
    tokenizer=hf_tokenizer,
    entropy_threshold=2.0,
    max_switches=3,
)
result = engine.generate_with_switch_thinking("Explain quantum computing:", max_new_tokens=128)
print(result.visible_text)
print(result.summary())
```

### AgentBenchSuite

```python
from src.evaluation import AgentBenchSuite

suite = AgentBenchSuite()
result = suite.full_evaluation(
    agent_trace=trace,
    ground_truth_tools=expected_tools,
    plan=declared_plan,
    actual_steps=executed_steps,
)
print(f"Composite Score: {result['composite_score']:.4f}")
```

## Mathematical Foundations

### SLERP (Spherical Linear Interpolation)

$$\text{Slerp}(p_0, p_1; t) = \frac{\sin((1-t)\theta)}{\sin\theta} p_0 + \frac{\sin(t\theta)}{\sin\theta} p_1$$

where $\theta = \arccos(\hat{p}_0 \cdot \hat{p}_1)$

### Shannon Entropy

$$H(X) = -\sum_{i=1}^{M} P(x_i) \log_2 P(x_i)$$

### Elo Rating

$$E = \frac{1}{1 + 10^{(R_{opponent} - R_{self}) / 400}}$$

## License

MIT License — See LICENSE for details.

## Contributing

Contributions welcome! Please open an issue or PR.
