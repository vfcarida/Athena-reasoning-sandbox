"""Athena Reasoning Sandbox — Main Orchestrator & CLI.

Provides a unified entry point that demonstrates the full capability stack of the
framework across two tiers:

    **Tier 1 — Offline (no GPU, no cloud credentials)**
        - AutonomousDataAgent: multi-turn FinOps SQL reflection with DuckDB backend.
        - RetrievalSearch:     hybrid BM25 + dense document retrieval via the executor.
        - CheckpointManager:   conversation state persistence and idempotency.

    **Tier 2 — ML demos (requires ``pip install -e '.[ml]'``)**
        - Phase 1a: From-scratch Transformer pretraining.
        - Phase 1b: Continued pretraining / domain adaptation config.
        - Phase 1c: LoRA / QLoRA fine-tuning configuration.
        - Phase 1d: Model merging (SLERP, TIES, DARE) + SwiReasoning.

Usage::

    # Run all offline (Tier 1) demos:
    python -m src.main

    # Run a specific demo group:
    python -m src.main --demo agent
    python -m src.main --demo retrieval
    python -m src.main --demo checkpoint
    python -m src.main --demo ml        # Tier 2, requires [ml] extras

    # List available demos:
    python -m src.main --list
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import uuid

# Ensure project root is on the module search path when invoked as a script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force UTF-8 output on Windows to prevent codec errors with box-drawing chars.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Terminal formatting helpers
# =============================================================================

class C:
    """ANSI escape codes for terminal colouring."""

    H = "\033[95m"
    B = "\033[94m"
    CN = "\033[96m"
    G = "\033[92m"
    Y = "\033[93m"
    R = "\033[91m"
    BD = "\033[1m"
    DM = "\033[2m"
    RS = "\033[0m"


def header(title: str) -> None:
    """Print a bold section header."""
    print(f"\n{C.BD}{C.CN}{'═' * 72}{C.RS}")
    print(f"{C.BD}{C.CN}  {title}{C.RS}")
    print(f"{C.BD}{C.CN}{'═' * 72}{C.RS}\n")


def sub(title: str) -> None:
    """Print a sub-section title."""
    print(f"\n{C.BD}{C.Y}  ── {title} ──{C.RS}\n")


def metric(name: str, value: object, colour: str = C.G) -> None:
    """Print a single metric row."""
    print(f"  {C.DM}│{C.RS} {name:<35} {colour}{value}{C.RS}")


def ok(msg: str) -> None:
    """Print a green success line."""
    print(f"\n  {C.G}✓ {msg}{C.RS}")


def warn(msg: str) -> None:
    """Print a yellow warning line."""
    print(f"\n  {C.Y}⚠ {msg}{C.RS}")


def fail(msg: str) -> None:
    """Print a red error line."""
    print(f"\n  {C.R}✗ {msg}{C.RS}")


# =============================================================================
# Demo — AutonomousDataAgent (Tier 1, offline)
# =============================================================================

def demo_agent() -> None:
    """Demonstrate the AutonomousDataAgent with multi-turn FinOps SQL reflection.

    Uses the DuckDB backend (Lane-1 offline) and seeds an in-memory table so the
    demo runs without any external service or AWS credentials.
    """
    header("DEMO: AutonomousDataAgent — Multi-Turn FinOps SQL Reflection")

    from src.athena.duckdb_client import DuckDBClient
    from src.engine.data_agent import AutonomousDataAgent

    sub("Seeding in-memory DuckDB table (orders)")
    # DuckDBClient creates its own in-memory connection; we seed via its .con attribute
    # so that the agent's query guard and executor share the same database state.
    client = DuckDBClient()
    client.con.execute(
        "CREATE TABLE orders (order_id INTEGER, amount DOUBLE, dt VARCHAR)"
    )
    client.con.execute(
        "INSERT INTO orders VALUES (1, 125.50, '2026-09-01'), (2, 230.00, '2026-09-01'), "
        "(3, 89.99, '2026-09-02')"
    )
    metric("Table", "orders")
    metric("Rows inserted", 3)
    metric("Partition column", "dt")
    agent = AutonomousDataAgent(backend="duckdb", duckdb_client=client, max_reflection_turns=3)

    sub("Turn 1 — Compliant query (partition filter present)")
    summary_ok = asyncio.run(
        agent.run(
            goal="Sum order amounts for partition 2026-09-01",
            initial_sql="SELECT SUM(amount) AS total FROM orders WHERE dt = '2026-09-01' LIMIT 100",
            required_partition_keys=["dt"],
        )
    )
    metric("Success", summary_ok.success)
    metric("SQL turns", summary_ok.turns)
    metric("Total time (ms)", f"{summary_ok.total_time_ms:.1f}")

    sub("Turn 2 — Self-correcting query (missing partition filter)")
    summary_fix = asyncio.run(
        agent.run(
            goal="Count all orders (agent must self-correct)",
            initial_sql="SELECT COUNT(*) FROM orders LIMIT 100",
            required_partition_keys=["dt"],
        )
    )
    metric("Success", summary_fix.success)
    metric("SQL turns (reflection)", summary_fix.turns)
    metric("Final SQL", summary_fix.final_sql or "")
    if summary_fix.turns > 1:
        metric("Reflection log", summary_fix.thinking_log[1][:80] + "...")
    ok("AutonomousDataAgent demo complete.")


# =============================================================================
# Demo — RetrievalSearch via ExecutiveEngineProcess (Tier 1, offline)
# =============================================================================

def demo_retrieval() -> None:
    """Demonstrate the retrieval_search tool via the full Parallax boundary.

    Indexes a small corpus, then executes a RETRIEVAL_SEARCH plan step through
    AgentLoop so the entire cognitive → executive path is exercised.
    """
    header("DEMO: RetrievalSearch — Hybrid BM25 + Dense RRF via Parallax Boundary")

    from src.engine.agent_loop import AgentLoop, AgentPlanner
    from src.engine.dispatcher import ParallaxToolDispatcher
    from src.engine.executor import ExecutiveEngineProcess
    from src.rag.retrieval_index import RetrievalIndex
    from src.reasoning.schemas import ActionType, ReasoningStep

    corpus = [
        {"doc_id": "doc-1", "content": "DuckDB is an in-process OLAP SQL database."},
        {"doc_id": "doc-2", "content": "FinOps governs cloud cost optimisation for data lake queries."},
        {"doc_id": "doc-3", "content": "Hybrid search fuses BM25 lexical and dense semantic rankings via RRF."},
        {"doc_id": "doc-4", "content": "The Parallax boundary separates cognitive LLM reasoning from tool execution."},
        {"doc_id": "doc-5", "content": "Athena is a serverless SQL engine for querying data in Amazon S3."},
    ]

    index = RetrievalIndex()
    index.index_documents(corpus)
    metric("Indexed documents", index.document_count)

    executor = ExecutiveEngineProcess(retrieval_index=index)
    dispatcher = ParallaxToolDispatcher(executor=executor)
    loop = AgentLoop(dispatcher=dispatcher)
    planner = AgentPlanner()

    sub("Executing RETRIEVAL_SEARCH plan step through AgentLoop")
    query = "serverless SQL query engine for data lake"
    step = ReasoningStep(
        step_number=1,
        rationale=f"Retrieve documents relevant to: '{query}'",
        action_type=ActionType.RETRIEVAL_SEARCH,
        tool_name="retrieval_search",
        tool_args={"operation": "search", "query": query, "top_k": 3},
    )
    plan = planner.create_plan(
        task_goal=f"Find documents about: {query}",
        steps=[step],
        plan_id=str(uuid.uuid4()),
    )
    result = asyncio.run(loop.run_plan(plan=plan))

    metric("Plan success", result.success)
    if result.final_output:
        hits = result.final_output.get("hits", [])
        metric("Top hits returned", len(hits))
        for i, hit in enumerate(hits, 1):
            metric(f"  Rank {i} ({hit['doc_id']})", f"score={hit['rrf_score']:.5f}  {hit['content'][:55]}...")
    ok("RetrievalSearch demo complete.")


# =============================================================================
# Demo — Checkpoint & Idempotency (Tier 1, offline)
# =============================================================================

def demo_checkpoint() -> None:
    """Demonstrate the ConversationCheckpointManager with idempotency keys."""
    header("DEMO: ConversationCheckpointManager — Idempotency & State Persistence")

    import tempfile

    from src.state.checkpoint_manager import (
        ConversationCheckpointManager,
        ConversationStateCheckpoint,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConversationCheckpointManager(storage_dir=tmpdir)

        sub("Creating and saving a conversation checkpoint")
        ck = ConversationStateCheckpoint(
            checkpoint_id=str(uuid.uuid4()),
            step_index=1,
            turn_history=[
                {"role": "user", "content": "Analyse Q3 2026 cloud spending."},
                {"role": "assistant", "content": "<think>Formulating FinOps query...</think>"},
            ],
            effect_ledger={"duckdb_query:orders_q3": {"status": "completed", "rows": 3}},
        )
        manager.save_checkpoint(ck)
        metric("Checkpoint ID", ck.checkpoint_id[:16] + "…")
        metric("Step index", ck.step_index)
        metric("Turn history entries", len(ck.turn_history))
        metric("Effect ledger entries", len(ck.effect_ledger))

        sub("Resuming from checkpoint (idempotency test)")
        restored = manager.restore_checkpoint(ck.checkpoint_id)
        metric("Restored step index", restored.step_index if restored else "NOT FOUND")
        metric("Turn history preserved", len(restored.turn_history) if restored else 0)

        sub("Idempotency: saving same checkpoint_id twice is safe")
        manager.save_checkpoint(ck)  # second save must not raise
        metric("Re-save result", "No error raised (idempotent)")

    ok("ConversationCheckpointManager demo complete.")


# =============================================================================
# Demo — ML Tier 2 (requires [ml] extras)
# =============================================================================

def demo_ml() -> None:
    """Demonstrate Tier-2 ML capabilities: pretraining, fine-tuning, and model merging.

    Requires ``pip install -e '.[ml]'``.  The demo gracefully skips if PyTorch is
    not installed rather than crashing the entire entry-point.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        warn(
            "PyTorch not installed.  Run 'pip install -e \\'.[ml]\\'' to enable ML demos."
        )
        return

    _demo_from_scratch()
    _demo_continued_pretraining()
    _demo_lora_finetuning()
    _demo_merging_and_reasoning()
    _demo_agent_bench_bridge()


def _demo_from_scratch() -> None:
    """Demonstrate from-scratch Transformer initialization and one training step."""

    header("PHASE 1a: From-Scratch Pretraining")

    from src.pretraining.from_scratch import PretrainingConfig, TextDataset, TransformerFromScratch

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
    metric("Parameters", f"{num_params:,} ({num_params / 1e6:.2f}M)")
    metric("Device", str(trainer.device))

    corpus = [
        "Language models process input and generate predictions token by token.",
        "Shannon entropy measures uncertainty in the output probability distribution.",
        "SLERP and TIES enable combining model specialisations through merging.",
        "LoRA fine-tuning drastically reduces the number of trainable parameters.",
        "4-bit quantisation allows running large models on limited hardware.",
    ] * 20

    tokenizer = trainer.build_tokenizer(corpus)
    dataset = TextDataset(corpus, tokenizer, max_length=64)
    history = trainer.train(dataset, model)

    if history["train_loss"]:
        metric("Final loss", f"{history['train_loss'][-1]:.4f}")
    ok("From-scratch pretraining demo complete.")


def _demo_continued_pretraining() -> None:
    """Demonstrate continued pretraining configuration (no model download)."""
    header("PHASE 1b: Continued Pretraining / Domain Adaptation")

    from src.pretraining.continued_pretraining import (
        ContinuedPretrainer,
        ContinuedPretrainingConfig,
    )

    config = ContinuedPretrainingConfig(
        base_model="gpt2",
        rope_scaling_type="linear",
        rope_scaling_factor=4.0,
        learning_rate=2e-5,
        max_seq_length=1024,
    )
    _pretrainer = ContinuedPretrainer(config)

    metric("Base model", config.base_model)
    metric("RoPE scaling", f"{config.rope_scaling_type} × {config.rope_scaling_factor}")
    metric("Original context", "2,048 tokens")
    metric("Extended context", f"{int(2048 * config.rope_scaling_factor):,} tokens")
    ok("Continued pretraining config demo complete.")


def _demo_lora_finetuning() -> None:
    """Demonstrate LoRA adapter configuration and SFT formatting."""
    header("PHASE 1c: Fine-Tuning — SFT + LoRA / QLoRA")

    from src.finetuning.lora_trainer import LoRAConfig
    from src.finetuning.sft_trainer import SFTConfig, SFTOrchestrator

    lora_config = LoRAConfig(
        model_name="gpt2",
        rank=16,
        alpha=32,
        dropout=0.05,
        target_modules=["c_attn", "c_proj"],
        quantize_4bit=False,
    )
    metric("LoRA rank (r)", lora_config.rank)
    metric("LoRA alpha (α)", lora_config.alpha)
    metric("Effective scaling", f"{lora_config.alpha / lora_config.rank:.1f}×")

    sft_config = SFTConfig(model_name="gpt2", dataset_format="alpaca")
    orchestrator = SFTOrchestrator(sft_config)
    sample = {
        "instruction": "Explain the concept of SLERP in model merging.",
        "input": "",
        "output": "SLERP performs geometric interpolation on the hypersurface of normalised "
                  "weight vectors, preserving the angular characteristics of parameter matrices.",
    }
    formatted = orchestrator.format_sample(sample)
    print(f"  {C.DM}│ Alpaca sample (first 120 chars):{C.RS}")
    print(f"  {C.DM}│{C.RS}   {C.B}{formatted[:120]}…{C.RS}")
    ok("Fine-tuning configuration demo complete.")


def _demo_merging_and_reasoning() -> None:
    """Demonstrate SLERP/TIES/DARE merging and SwiReasoning simulation."""
    import torch

    header("PHASE 1d: Model Merging + SwiReasoning")

    from src.merging.merge_operators import TensorMergeOperators
    from src.reasoning.swi_reasoning import SwiReasoningSimulator
    from src.utils.metrics import overthinking_index

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    dim = 512
    base = torch.randn(dim, dim, device=device) * 0.02
    model_a = base + torch.randn(dim, dim, device=device) * 0.01
    model_b = base + torch.randn(dim, dim, device=device) * 0.01

    sub("SLERP — Spherical Linear Interpolation")
    for t in [0.0, 0.5, 1.0]:
        merged = TensorMergeOperators.slerp(model_a, model_b, t=t)
        cos_a = torch.nn.functional.cosine_similarity(
            merged.flatten().unsqueeze(0), model_a.flatten().unsqueeze(0),
        ).item()
        metric(f"t={t:.1f}", f"cos(A)={cos_a:.6f}")

    sub("SwiReasoning — Entropy-Guided Adaptive Inference")
    sim = SwiReasoningSimulator(
        vocab_size=32000, entropy_threshold=2.0,
        max_switches=3, max_thinking_tokens=8, seed=42,
    )
    result = sim.simulate("Explain quantum entanglement:", num_steps=20)
    summary = result.summary()
    metric("Total tokens", summary["total_tokens"])
    metric("Thinking tokens", summary["thinking_tokens"], C.Y)
    ot = overthinking_index(summary["thinking_tokens"], summary["total_tokens"])
    metric("Efficiency", f"{ot['efficiency_score']:.4f}", C.G)
    ok("Merging + SwiReasoning demo complete.")


def _demo_agent_bench_bridge() -> None:
    """Show Agent-Bench bridge status and available metrics."""
    header("BRIDGE: Agent-Bench Integration")

    from src.bridge.agent_bench_bridge import AgentBenchBridge, BridgeConfig

    bridge = AgentBenchBridge(BridgeConfig(
        agent_bench_path=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "Agent-Bench")
        ),
    ))
    status = "✓ CONNECTED" if bridge.is_available else "⚠ NOT INSTALLED"
    metric("Agent-Bench", status, C.G if bridge.is_available else C.Y)
    ok("Agent-Bench bridge status check complete.")


# =============================================================================
# CLI
# =============================================================================

DEMO_MAP: dict[str, tuple[str, object]] = {
    "agent": ("AutonomousDataAgent (multi-turn SQL reflection, offline)", demo_agent),
    "retrieval": ("RetrievalSearch (hybrid BM25+dense via Parallax boundary, offline)", demo_retrieval),
    "checkpoint": ("ConversationCheckpointManager (idempotency, offline)", demo_checkpoint),
    "ml": ("ML demos: pretraining / fine-tuning / merging (requires [ml] extras)", demo_ml),
}


def _print_banner() -> None:
    print(f"\n{C.BD}{C.CN}")
    print("  ╔═══════════════════════════════════════════════════════════════╗")
    print("  ║                                                               ║")
    print("  ║      ██████╗ ████████╗██╗  ██╗███████╗███╗  ██╗ █████╗      ║")
    print("  ║     ██╔══██╗╚══██╔══╝██║  ██║██╔════╝████╗ ██║██╔══██╗     ║")
    print("  ║     ███████║   ██║   ███████║█████╗  ██╔██╗██║███████║     ║")
    print("  ║     ██╔══██║   ██║   ██╔══██║██╔══╝  ██║╚████║██╔══██║     ║")
    print("  ║     ██║  ██║   ██║   ██║  ██║███████╗██║ ╚███║██║  ██║     ║")
    print("  ║     ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚══╝╚═╝  ╚═╝     ║")
    print("  ║                                                               ║")
    print("  ║     R E A S O N I N G   S A N D B O X   v0.1.0               ║")
    print("  ║     FinOps-Governed Autonomous Data Lake Agent Framework      ║")
    print("  ║                                                               ║")
    print("  ╚═══════════════════════════════════════════════════════════════╝")
    print(f"{C.RS}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description=(
            "Athena Reasoning Sandbox — demonstration entry point.\n\n"
            "Without --demo, runs all Tier-1 offline demos (agent, retrieval, checkpoint).\n"
            "Use --demo ml to additionally run Tier-2 ML demos (requires [ml] extras)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--demo",
        choices=list(DEMO_MAP.keys()),
        metavar="DEMO",
        help=(
            "Run a specific demo group. "
            "Choices: " + ", ".join(DEMO_MAP.keys()) + "."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available demo groups and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code (0 = success, 1 = error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print("\nAvailable demos:\n")
        for key, (desc, _) in DEMO_MAP.items():
            print(f"  {C.BD}--demo {key:<12}{C.RS}  {desc}")
        print()
        return 0

    _print_banner()

    demos_to_run: list[tuple[str, object]]
    if args.demo:
        demos_to_run = [(args.demo, DEMO_MAP[args.demo][1])]
    else:
        # Default: run all Tier-1 offline demos
        demos_to_run = [
            ("agent", demo_agent),
            ("retrieval", demo_retrieval),
            ("checkpoint", demo_checkpoint),
        ]

    exit_code = 0
    for name, fn in demos_to_run:
        try:
            fn()  # type: ignore[call-arg,operator]
        except Exception as exc:  # noqa: BLE001
            fail(f"Demo '{name}' raised an unexpected error: {exc}")
            import traceback
            traceback.print_exc()
            exit_code = 1

    if exit_code == 0:
        print(f"\n{C.BD}{C.G}")
        print("  ═══════════════════════════════════════════════════════════════")
        print("  ✓  All demonstrations completed successfully.")
        print(f"  ═══════════════════════════════════════════════════════════════{C.RS}\n")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
