# Athena Reasoning Sandbox

> **Scientific Framework for Neural Model Experimentation**
> Pretraining • Fine-Tuning • Merging • Adaptive Inference

---

## Overview

Athena Reasoning Sandbox is a research-grade Python framework for the **full lifecycle of language model development**. It implements the four pathways of model specialization described in the corporate AI strategy framework:

| Path | Module | Description |
|------|--------|-------------|
| **1a** From-Scratch | `src/pretraining/from_scratch.py` | Random-init Transformer + CLM pretraining loop |
| **1b** Continued | `src/pretraining/continued_pretraining.py` | Domain adaptation with RoPE context extension |
| **1c** Fine-Tuning | `src/finetuning/` | SFT + LoRA + QLoRA (4-bit NF4) |
| **1d** Merging | `src/merging/` + `src/reasoning/` | SLERP, TIES, DARE + SwiReasoning inference |

**Evaluation** is handled by the companion [Agent-Bench](https://github.com/vfcarida/Agent-Bench) framework, connected via the `src/bridge/` integration module.

## Architecture

```
Athena-reasoning-sandbox/
├── configs/
│   ├── merge_config.yaml              # SLERP/TIES/DARE parameters
│   ├── pretraining_config.yaml        # Architecture + training hyperparams
│   └── finetuning_config.yaml         # SFT, LoRA, QLoRA settings
├── src/
│   ├── main.py                        # Orchestrator — demo all 4 pathways
│   ├── pretraining/
│   │   ├── from_scratch.py            # Transformer init + CLM training
│   │   └── continued_pretraining.py   # Domain adaptation + RoPE scaling
│   ├── finetuning/
│   │   ├── sft_trainer.py             # Supervised Fine-Tuning (Alpaca/ChatML)
│   │   └── lora_trainer.py            # LoRA / QLoRA adapters (PEFT)
│   ├── merging/
│   │   ├── merge_operators.py         # SLERP, TIES, DARE tensor fusion
│   │   └── merge_pipeline.py          # YAML-driven merge orchestrator
│   ├── reasoning/
│   │   └── swi_reasoning.py           # Entropy-guided adaptive inference
│   ├── bridge/
│   │   └── agent_bench_bridge.py      # Agent-Bench evaluation connector
│   └── utils/
│       └── metrics.py                 # Shannon entropy, Elo, OT index
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/vfcarida/Athena-reasoning-sandbox.git
cd Athena-reasoning-sandbox

python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt

# Optional: QLoRA support (GPU only)
pip install bitsandbytes>=0.41.0

# Optional: Connect to Agent-Bench for evaluation
pip install -e ../Agent-Bench
```

**Requirements**: Python 3.10+, PyTorch 2.0+, Transformers 4.35+

## Quick Start

```bash
python -m src.main
```

## Development Pathways

### 1a — From-Scratch Pretraining

```python
from src.pretraining import TransformerFromScratch
from src.pretraining.from_scratch import PretrainingConfig, TextDataset

config = PretrainingConfig(hidden_size=768, num_hidden_layers=12)
trainer = TransformerFromScratch(config)
model = trainer.build_model()
tokenizer = trainer.build_tokenizer(corpus_texts)
dataset = TextDataset(corpus_texts, tokenizer, max_length=2048)
history = trainer.train(dataset, model)
```

### 1b — Continued Pretraining

```python
from src.pretraining import ContinuedPretrainer
from src.pretraining.continued_pretraining import ContinuedPretrainingConfig

config = ContinuedPretrainingConfig(
    base_model="meta-llama/Llama-2-7b-hf",
    rope_scaling_type="yarn",
    rope_scaling_factor=4.0,
)
pretrainer = ContinuedPretrainer(config)
model, tokenizer = pretrainer.load_base_model()
dataset = pretrainer.prepare_domain_dataset(domain_texts, tokenizer)
history = pretrainer.train(model, dataset)
```

### 1c — LoRA / QLoRA Fine-Tuning

```python
from src.finetuning import LoRAFineTuner, SFTOrchestrator
from src.finetuning.lora_trainer import LoRAConfig
from src.finetuning.sft_trainer import SFTConfig

# LoRA adapter injection
lora_config = LoRAConfig(rank=16, alpha=32, quantize_4bit=True)
tuner = LoRAFineTuner(lora_config)
model, tokenizer = tuner.prepare_model()

# SFT training
sft = SFTOrchestrator(SFTConfig(model_name="gpt2"))
sft.train(train_data, model=model, tokenizer=tokenizer)

# Merge adapters back to standalone model
tuner.merge_and_save(model, tokenizer, "./merged_model")
```

### 1d — Model Merging + SwiReasoning

```python
from src.merging import TensorMergeOperators
from src.reasoning import SwiReasoningEngine

# Merge specialized models
merged = TensorMergeOperators.slerp(weights_a, weights_b, t=0.5)
merged = TensorMergeOperators.ties_merge(base, [model_a, model_b], threshold=0.2)

# Entropy-guided inference
engine = SwiReasoningEngine(model, tokenizer, entropy_threshold=2.0)
result = engine.generate_with_switch_thinking("Explain quantum computing:")
```

## Agent-Bench Integration

Models trained in this sandbox can be evaluated using the [Agent-Bench](https://github.com/vfcarida/Agent-Bench) framework:

```python
from src.bridge import AgentBenchBridge

bridge = AgentBenchBridge()
adapter = bridge.create_model_adapter(model, tokenizer)
bridge.export_for_evaluation(model, tokenizer, "./export")
results = bridge.run_benchmark("pix_basic_v1")
```

## Mathematical Foundations

### SLERP
$$\text{Slerp}(p_0, p_1; t) = \frac{\sin((1-t)\theta)}{\sin\theta} p_0 + \frac{\sin(t\theta)}{\sin\theta} p_1$$

### Shannon Entropy (SwiReasoning)
$$H(X) = -\sum_{i=1}^{M} P(x_i) \log_2 P(x_i)$$

### LoRA Decomposition
$$W' = W + \frac{\alpha}{r} \cdot B \cdot A, \quad A \in \mathbb{R}^{d \times r}, B \in \mathbb{R}^{r \times d}$$

## License

MIT License
