"""Athena Reasoning Sandbox — Main Orchestrator.

Demonstrates all four model development pathways:
    Phase 1a: From-scratch pretraining (tiny model demo)
    Phase 1b: Continued pretraining / domain adaptation (config demo)
    Phase 1c: LoRA/QLoRA fine-tuning (adapter injection demo)
    Phase 1d: Model merging (SLERP, TIES, DARE) + SwiReasoning

Run directly:
    python -m src.main
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

import torch


# ═════════════════════════════════════════════════════════════════════════════
# Terminal formatting helpers
# ═════════════════════════════════════════════════════════════════════════════
class C:
    """ANSI escape codes."""
    H = "\033[95m"; B = "\033[94m"; CN = "\033[96m"; G = "\033[92m"
    Y = "\033[93m"; R = "\033[91m"; BD = "\033[1m"; DM = "\033[2m"
    RS = "\033[0m"


def header(t: str) -> None:
    print(f"\n{C.BD}{C.CN}{'═' * 72}{C.RS}")
    print(f"{C.BD}{C.CN}  {t}{C.RS}")
    print(f"{C.BD}{C.CN}{'═' * 72}{C.RS}\n")


def sub(t: str) -> None:
    print(f"\n{C.BD}{C.Y}  ── {t} ──{C.RS}\n")


def metric(n: str, v: object, c: str = C.G) -> None:
    print(f"  {C.DM}│{C.RS} {n:<35} {c}{v}{C.RS}")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1a: From-Scratch Pretraining Demo
# ═════════════════════════════════════════════════════════════════════════════

def demo_from_scratch() -> None:
    """Demonstrate from-scratch Transformer initialization and one training step."""
    header("PHASE 1a: From-Scratch Pretraining")

    from src.pretraining.from_scratch import TransformerFromScratch, PretrainingConfig, TextDataset

    config = PretrainingConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=512,
        max_position_embeddings=256,
        epochs=1,
        batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        log_steps=1,
        save_steps=9999,
    )

    trainer = TransformerFromScratch(config)
    model = trainer.build_model()

    num_params = sum(p.numel() for p in model.parameters())
    metric("Architecture", f"GPT-2 (tiny: {config.num_hidden_layers}L, {config.hidden_size}H)")
    metric("Parameters", f"{num_params:,} ({num_params/1e6:.2f}M)")
    metric("Device", str(trainer.device))

    sub("Tokenizer Training")
    corpus = [
        "O modelo de linguagem processa texto de entrada e gera previsões token a token.",
        "A entropia de Shannon mede a incerteza na distribuição de probabilidade de saída.",
        "Técnicas de fusão de modelos como SLERP e TIES permitem combinar especializações.",
        "O ajuste fino com LoRA reduz drasticamente o número de parâmetros treináveis.",
        "A quantização em 4 bits permite executar modelos grandes em hardware limitado.",
    ] * 20  # Repeat for sufficient tokenizer training data

    tokenizer = trainer.build_tokenizer(corpus)
    metric("Vocabulary size", len(tokenizer))

    sub("Training Loop (1 epoch, tiny model)")
    dataset = TextDataset(corpus, tokenizer, max_length=64)
    metric("Training samples", len(dataset))

    history = trainer.train(dataset, model)

    if history["train_loss"]:
        metric("Final loss", f"{history['train_loss'][-1]:.4f}")
    metric("Training steps", len(history["train_loss"]))

    print(f"\n  {C.G}✓ From-scratch pretraining demo complete.{C.RS}")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1b: Continued Pretraining Demo
# ═════════════════════════════════════════════════════════════════════════════

def demo_continued_pretraining() -> None:
    """Demonstrate continued pretraining configuration (no model download)."""
    header("PHASE 1b: Continued Pretraining / Domain Adaptation")

    from src.pretraining.continued_pretraining import ContinuedPretrainer, ContinuedPretrainingConfig

    config = ContinuedPretrainingConfig(
        base_model="gpt2",
        rope_scaling_type="linear",
        rope_scaling_factor=4.0,
        learning_rate=2e-5,
        max_seq_length=1024,
    )

    pretrainer = ContinuedPretrainer(config)

    sub("Configuration Summary")
    metric("Base model", config.base_model)
    metric("RoPE scaling", f"{config.rope_scaling_type} × {config.rope_scaling_factor}")
    metric("Learning rate", f"{config.learning_rate:.2e}")
    metric("Max sequence length", config.max_seq_length)
    metric("Gradient checkpointing", config.gradient_checkpointing)
    metric("Freeze embeddings", config.freeze_embeddings)

    sub("Context Window Extension")
    # Demonstrate the concept without loading a large model
    original_ctx = 2048
    extended_ctx = int(original_ctx * config.rope_scaling_factor)
    metric("Original context", f"{original_ctx:,} tokens")
    metric("Extended context", f"{extended_ctx:,} tokens")
    metric("Extension method", f"RoPE {config.rope_scaling_type} scaling")

    print(f"\n  {C.DM}│ Note: Actual model loading requires downloading from HF Hub.{C.RS}")
    print(f"  {C.DM}│ Use: pretrainer.load_base_model() to load and train.{C.RS}")
    print(f"\n  {C.G}✓ Continued pretraining config demo complete.{C.RS}")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1c: LoRA / QLoRA Fine-Tuning Demo
# ═════════════════════════════════════════════════════════════════════════════

def demo_lora_finetuning() -> None:
    """Demonstrate LoRA adapter configuration and SFT formatting."""
    header("PHASE 1c: Fine-Tuning — SFT + LoRA / QLoRA")

    from src.finetuning.sft_trainer import SFTOrchestrator, SFTConfig
    from src.finetuning.lora_trainer import LoRAFineTuner, LoRAConfig

    sub("LoRA Configuration")
    lora_config = LoRAConfig(
        model_name="gpt2",
        rank=16,
        alpha=32,
        dropout=0.05,
        target_modules=["c_attn", "c_proj"],  # GPT-2 module names
        quantize_4bit=False,
    )

    metric("LoRA rank (r)", lora_config.rank)
    metric("LoRA alpha (α)", lora_config.alpha)
    metric("Effective scaling (α/r)", f"{lora_config.alpha / lora_config.rank:.1f}×")
    metric("Dropout", lora_config.dropout)
    metric("Target modules", ", ".join(lora_config.target_modules))
    metric("4-bit quantization (QLoRA)", lora_config.quantize_4bit)

    sub("QLoRA Configuration (reference)")
    qlora_config = LoRAConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        rank=64,
        alpha=128,
        quantize_4bit=True,
        quantization_type="nf4",
        compute_dtype="bfloat16",
        use_double_quant=True,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    metric("Model", qlora_config.model_name)
    metric("Quantization type", qlora_config.quantization_type)
    metric("Compute dtype", qlora_config.compute_dtype)
    metric("Double quantization", qlora_config.use_double_quant)
    metric("LoRA rank", qlora_config.rank)
    metric("Target modules", f"{len(qlora_config.target_modules)} attention + FFN modules")

    sub("SFT Dataset Formatting")
    sft_config = SFTConfig(model_name="gpt2", dataset_format="alpaca")
    orchestrator = SFTOrchestrator(sft_config)

    sample = {
        "instruction": "Explique o conceito de SLERP em fusão de modelos.",
        "input": "",
        "output": "SLERP (Spherical Linear Interpolation) realiza interpolação geométrica "
                  "na hipersuperfície de vetores de pesos normalizados, preservando as "
                  "características angulares das matrizes de parâmetros.",
    }

    formatted = orchestrator.format_sample(sample)
    print(f"  {C.DM}│ Alpaca format:{C.RS}")
    for line in formatted.split("\n"):
        print(f"  {C.DM}│{C.RS}   {C.B}{line}{C.RS}")

    print(f"\n  {C.G}✓ Fine-tuning configuration demo complete.{C.RS}")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1d: Model Merging + SwiReasoning
# ═════════════════════════════════════════════════════════════════════════════

def demo_merging_and_reasoning() -> None:
    """Demonstrate SLERP/TIES/DARE merging and SwiReasoning simulation."""
    header("PHASE 1d: Model Merging + SwiReasoning")

    from src.merging.merge_operators import TensorMergeOperators
    from src.reasoning.swi_reasoning import SwiReasoningSimulator
    from src.utils.metrics import overthinking_index

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Synthetic tensors
    dim = 512
    base = torch.randn(dim, dim, device=device) * 0.02
    model_a = base + torch.randn(dim, dim, device=device) * 0.01
    model_b = base + torch.randn(dim, dim, device=device) * 0.01

    sub("SLERP — Spherical Linear Interpolation")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        merged = TensorMergeOperators.slerp(model_a, model_b, t=t)
        cos_a = torch.nn.functional.cosine_similarity(
            merged.flatten().unsqueeze(0), model_a.flatten().unsqueeze(0),
        ).item()
        cos_b = torch.nn.functional.cosine_similarity(
            merged.flatten().unsqueeze(0), model_b.flatten().unsqueeze(0),
        ).item()
        metric(f"t={t:.2f}", f"cos(A)={cos_a:.6f}  cos(B)={cos_b:.6f}")

    sub("TIES-Merging")
    model_c = base + torch.randn(dim, dim, device=device) * 0.01
    for thr in [0.1, 0.2, 0.3]:
        merged = TensorMergeOperators.ties_merge(base, [model_a, model_b, model_c], threshold=thr)
        delta = (merged - base).norm().item()
        metric(f"threshold={thr:.1f}", f"Δ‖base‖={delta:.6f}")

    sub("DARE — Drop And REscale")
    for dr in [0.1, 0.3, 0.5]:
        merged = TensorMergeOperators.dare_drop_and_rescale(base, model_a, drop_rate=dr, seed=42)
        delta = (merged - base).norm().item()
        metric(f"drop_rate={dr:.1f}", f"Δ‖base‖={delta:.4f}")

    sub("SwiReasoning — Entropy-Guided Adaptive Inference")
    sim = SwiReasoningSimulator(
        vocab_size=32000, entropy_threshold=2.0,
        max_switches=3, max_thinking_tokens=8, seed=42,
    )
    result = sim.simulate("Explain quantum entanglement:", num_steps=20)
    summary = result.summary()

    metric("Total tokens", summary["total_tokens"])
    metric("Visible tokens", summary["visible_tokens"])
    metric("Thinking tokens", summary["thinking_tokens"], C.Y)
    metric("Mode switches", summary["switch_count"])
    metric("Avg entropy", f"{summary['entropy_avg']:.4f} bits")

    ot = overthinking_index(summary["thinking_tokens"], summary["total_tokens"])
    ot_c = C.G if not ot["is_overthinking"] else C.R
    metric("Efficiency", f"{ot['efficiency_score']:.4f}", ot_c)

    print(f"\n  {C.G}✓ Merging + SwiReasoning demo complete.{C.RS}")


# ═════════════════════════════════════════════════════════════════════════════
# Agent-Bench Bridge Status
# ═════════════════════════════════════════════════════════════════════════════

def demo_agent_bench_bridge() -> None:
    """Show Agent-Bench bridge status and available metrics."""
    header("BRIDGE: Agent-Bench Integration")

    from src.bridge.agent_bench_bridge import AgentBenchBridge, BridgeConfig

    bridge = AgentBenchBridge(BridgeConfig(
        agent_bench_path=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "Agent-Bench")
        ),
    ))

    sub("Connection Status")
    status = "✓ CONNECTED" if bridge.is_available else "⚠ NOT INSTALLED"
    status_c = C.G if bridge.is_available else C.Y
    metric("Agent-Bench", status, status_c)
    metric("Path", bridge.config.agent_bench_path)

    sub("Available Evaluation Metrics")
    metrics_ref = bridge.get_agent_bench_metrics()
    for name, desc in metrics_ref.items():
        metric(name, desc[:55] + "..." if len(desc) > 55 else desc, C.B)

    if not bridge.is_available:
        print(f"\n  {C.Y}  To enable evaluation, install Agent-Bench:{C.RS}")
        print(f"  {C.DM}  pip install -e ../Agent-Bench{C.RS}")

    print(f"\n  {C.G}✓ Agent-Bench bridge status check complete.{C.RS}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run all demonstration pipelines."""
    print(f"\n{C.BD}{C.CN}")
    print("  ╔═══════════════════════════════════════════════════════════════╗")
    print("  ║                                                             ║")
    print("  ║      █████╗ ████████╗██╗  ██╗███████╗███╗   ██╗ █████╗     ║")
    print("  ║     ██╔══██╗╚══██╔══╝██║  ██║██╔════╝████╗  ██║██╔══██╗    ║")
    print("  ║     ███████║   ██║   ███████║█████╗  ██╔██╗ ██║███████║    ║")
    print("  ║     ██╔══██║   ██║   ██╔══██║██╔══╝  ██║╚██╗██║██╔══██║    ║")
    print("  ║     ██║  ██║   ██║   ██║  ██║███████╗██║ ╚████║██║  ██║    ║")
    print("  ║     ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝    ║")
    print("  ║                                                             ║")
    print("  ║     R E A S O N I N G   S A N D B O X   v0.1.0             ║")
    print("  ║     Neural Model Experimentation Framework                  ║")
    print("  ║                                                             ║")
    print("  ╚═══════════════════════════════════════════════════════════════╝")
    print(f"{C.RS}")

    try:
        demo_from_scratch()
        demo_continued_pretraining()
        demo_lora_finetuning()
        demo_merging_and_reasoning()
        demo_agent_bench_bridge()
    except Exception as e:
        print(f"\n  {C.R}✗ Error: {e}{C.RS}")
        import traceback
        traceback.print_exc()
        raise

    print(f"\n{C.BD}{C.G}")
    print("  ═══════════════════════════════════════════════════════════════")
    print("  ✓  All demonstrations completed successfully.")
    print("  ═══════════════════════════════════════════════════════════════")
    print(f"{C.RS}")


if __name__ == "__main__":
    main()
