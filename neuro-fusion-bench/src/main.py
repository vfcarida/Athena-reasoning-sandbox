"""NeuroFusionBench — Main Orchestrator.

This module demonstrates the complete framework by executing three integrated
demonstration pipelines:

1. **Model Merging Demo**: Synthetic tensor fusion using SLERP, TIES, and DARE.
2. **SwiReasoning Demo**: Entropy-guided adaptive inference simulation.
3. **Agent-Bench Demo**: Agent trace evaluation with formatted metric output.

Run directly:
    python -m src.main

Or from the project root:
    python src/main.py
"""

from __future__ import annotations

import json
import sys
import os

# Ensure the project root is on the path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force UTF-8 output encoding on Windows to support Unicode box-drawing chars
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

import torch

from src.merging.merge_operators import TensorMergeOperators
from src.reasoning.swi_reasoning import SwiReasoningEngine, SwiReasoningSimulator
from src.evaluation.agent_bench import AgentBenchSuite
from src.evaluation.judge import LLMJudge
from src.utils.metrics import shannon_entropy, elo_rating, overthinking_index


# ═════════════════════════════════════════════════════════════════════════════
# ANSI Color Codes for Terminal Output
# ═════════════════════════════════════════════════════════════════════════════
class Colors:
    """ANSI escape codes for colorful terminal output."""
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def print_header(title: str) -> None:
    """Print a formatted section header."""
    width = 72
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'═' * width}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'═' * width}{Colors.RESET}\n")


def print_subheader(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}  ── {title} ──{Colors.RESET}\n")


def print_metric(name: str, value: object, color: str = Colors.GREEN) -> None:
    """Print a formatted metric line."""
    print(f"  {Colors.DIM}│{Colors.RESET} {name:<30} {color}{value}{Colors.RESET}")


# ═════════════════════════════════════════════════════════════════════════════
# DEMO 1: Model Merging
# ═════════════════════════════════════════════════════════════════════════════

def demo_model_merging() -> None:
    """Demonstrate SLERP, TIES-Merging, and DARE on synthetic tensors."""
    print_header("PHASE 1: Neural Weight Fusion Engine")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  {Colors.DIM}Device: {device}{Colors.RESET}")

    # Create synthetic weight tensors simulating model parameters
    torch.manual_seed(42)
    dim = 1024

    base = torch.randn(dim, dim, device=device) * 0.02
    model_a = base + torch.randn(dim, dim, device=device) * 0.01  # Fine-tuned model A
    model_b = base + torch.randn(dim, dim, device=device) * 0.01  # Fine-tuned model B
    model_c = base + torch.randn(dim, dim, device=device) * 0.01  # Fine-tuned model C

    # ─── SLERP ───────────────────────────────────────────────────────
    print_subheader("SLERP — Spherical Linear Interpolation")

    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        merged = TensorMergeOperators.slerp(model_a, model_b, t=t)

        # Compute cosine similarity with both source tensors
        cos_sim_a = torch.nn.functional.cosine_similarity(
            merged.flatten().unsqueeze(0),
            model_a.flatten().unsqueeze(0),
        ).item()
        cos_sim_b = torch.nn.functional.cosine_similarity(
            merged.flatten().unsqueeze(0),
            model_b.flatten().unsqueeze(0),
        ).item()

        print_metric(
            f"t={t:.2f}",
            f"cos(A)={cos_sim_a:.6f}  cos(B)={cos_sim_b:.6f}  "
            f"‖merged‖={merged.norm().item():.4f}",
        )

    # ─── TIES-Merging ────────────────────────────────────────────────
    print_subheader("TIES-Merging — Trim, Elect Sign & Merge")

    for threshold in [0.1, 0.2, 0.3, 0.5]:
        merged = TensorMergeOperators.ties_merge(
            base, [model_a, model_b, model_c], threshold=threshold,
        )
        delta_norm = (merged - base).norm().item()
        cos_sim_base = torch.nn.functional.cosine_similarity(
            merged.flatten().unsqueeze(0),
            base.flatten().unsqueeze(0),
        ).item()
        print_metric(
            f"threshold={threshold:.1f}",
            f"Δ‖base‖={delta_norm:.6f}  cos(base)={cos_sim_base:.6f}",
        )

    # ─── DARE ────────────────────────────────────────────────────────
    print_subheader("DARE — Drop And REscale")

    for drop_rate in [0.1, 0.3, 0.5, 0.7]:
        merged = TensorMergeOperators.dare_drop_and_rescale(
            base, model_a, drop_rate=drop_rate, seed=42,
        )
        delta_norm = (merged - base).norm().item()
        original_delta_norm = (model_a - base).norm().item()
        scale_ratio = delta_norm / max(original_delta_norm, 1e-12)

        print_metric(
            f"drop_rate={drop_rate:.1f}",
            f"Δ‖base‖={delta_norm:.4f}  "
            f"scale_ratio={scale_ratio:.4f}  "
            f"(original Δ={original_delta_norm:.4f})",
        )

    print(f"\n  {Colors.GREEN}✓ Model merging demo complete.{Colors.RESET}")


# ═════════════════════════════════════════════════════════════════════════════
# DEMO 2: SwiReasoning
# ═════════════════════════════════════════════════════════════════════════════

def demo_swi_reasoning() -> None:
    """Demonstrate entropy-guided adaptive inference with the simulator."""
    print_header("PHASE 2: SwiReasoning — Adaptive Inference Engine")

    sim = SwiReasoningSimulator(
        vocab_size=32000,
        entropy_threshold=2.0,
        max_switches=3,
        max_thinking_tokens=8,
        seed=42,
    )

    result = sim.simulate(
        prompt="Explain the implications of quantum entanglement on information theory:",
        num_steps=30,
    )

    # Display entropy trace with mode indicators
    print_subheader("Entropy Trace (per-token)")

    entropy_hist = result.state.entropy_history
    mode_labels: list[str] = []
    current_mode = "EXPLICIT"
    switch_count = 0

    for i, h in enumerate(entropy_hist):
        # Reconstruct mode from entropy values
        if current_mode == "EXPLICIT" and h > sim.entropy_threshold and switch_count < sim.max_switches:
            current_mode = "LATENT"
            switch_count += 1
        elif current_mode == "LATENT" and h < sim.entropy_threshold:
            current_mode = "EXPLICIT"

        bar_len = int(h * 8)
        bar = "█" * bar_len + "░" * (20 - bar_len)

        mode_color = Colors.RED if current_mode == "LATENT" else Colors.GREEN
        mode_tag = f"{mode_color}{'🧠 THINK' if current_mode == 'LATENT' else '💬 SPEAK'}{Colors.RESET}"

        print(
            f"  {Colors.DIM}│{Colors.RESET} "
            f"Step {i:>2d}  "
            f"H={h:>6.3f} bits  "
            f"{Colors.BLUE}{bar}{Colors.RESET}  "
            f"{mode_tag}"
        )

    # Summary statistics
    print_subheader("Generation Summary")
    summary = result.summary()
    print_metric("Total tokens", summary["total_tokens"])
    print_metric("Visible tokens", summary["visible_tokens"])
    print_metric("Thinking tokens", summary["thinking_tokens"], Colors.YELLOW)
    print_metric("Mode switches", summary["switch_count"])
    print_metric("Overthinking ratio", f"{summary['overthinking_ratio']:.2%}")
    print_metric("Avg entropy", f"{summary['entropy_avg']:.4f} bits")
    print_metric("Max entropy", f"{summary['entropy_max']:.4f} bits")
    print_metric("Min entropy", f"{summary['entropy_min']:.4f} bits")

    # Overthinking analysis
    ot = overthinking_index(
        thinking_tokens=summary["thinking_tokens"],
        total_tokens=summary["total_tokens"],
    )
    ot_color = Colors.GREEN if not ot["is_overthinking"] else Colors.RED
    print_metric("Efficiency score", f"{ot['efficiency_score']:.4f}", ot_color)
    print_metric(
        "Overthinking status",
        f"{'⚠ OVERTHINKING' if ot['is_overthinking'] else '✓ WITHIN LIMITS'}",
        ot_color,
    )

    # Show output with think markers
    print_subheader("Generated Output (with think markers)")
    output = result.output_text
    # Colorize think markers
    output = output.replace(
        "<think>",
        f"{Colors.RED}{Colors.BOLD}<think>{Colors.RESET}{Colors.DIM}",
    )
    output = output.replace(
        "</think>",
        f"{Colors.RESET}{Colors.RED}{Colors.BOLD}</think>{Colors.RESET}",
    )
    print(f"  {Colors.DIM}│{Colors.RESET} {output}")

    print(f"\n  {Colors.GREEN}✓ SwiReasoning demo complete.{Colors.RESET}")


# ═════════════════════════════════════════════════════════════════════════════
# DEMO 3: Agent-Bench
# ═════════════════════════════════════════════════════════════════════════════

def demo_agent_bench() -> None:
    """Demonstrate Agent-Bench evaluation with synthetic agent traces."""
    print_header("PHASE 3: Agent-Bench — Scientific Validation Suite")

    suite = AgentBenchSuite()
    judge = LLMJudge(mode="offline")

    # ─── Scenario: Web Navigation ────────────────────────────────────
    print_subheader("Scenario: Web Navigation Task")

    ground_truth = ["search", "click", "scroll", "click", "type_text"]
    agent_trace = [
        {"tool": "search", "params": {"query": "best restaurants"}},
        {"tool": "click", "params": {"element": "result_1"}},
        {"tool": "scroll", "params": {"direction": "down"}},
        {"tool": "scroll", "params": {"direction": "down"}},  # redundant
        {"tool": "click", "params": {"element": "menu_link"}},
        {"tool": "type_text", "params": {"text": "reservation for 2"}},
    ]

    tool_result = suite.evaluate_tool_usage(agent_trace, ground_truth)

    print_metric("Tool Correctness", f"{tool_result.tool_correctness:.4f}")
    print_metric("Tool Efficiency", f"{tool_result.tool_efficiency:.4f}")
    print_metric(
        "Redundancy Rate",
        f"{tool_result.redundancy_rate:.4f}",
        Colors.YELLOW if tool_result.redundancy_rate > 0 else Colors.GREEN,
    )
    print_metric("Parameter Accuracy", f"{tool_result.parameter_accuracy:.4f}")

    # ─── Plan Adherence ──────────────────────────────────────────────
    print_subheader("Plan Adherence Analysis")

    plan = [
        "search_for_restaurants",
        "select_top_result",
        "read_reviews",
        "navigate_to_menu",
        "make_reservation",
    ]
    actual_steps = [
        "search_for_restaurants",
        "select_top_result",
        "navigate_to_menu",
        "check_prices",          # extra step
        "make_reservation",
    ]

    adherence = suite.evaluate_plan_adherence(plan, actual_steps)

    print_metric("Adherence Score", f"{adherence.adherence_score:.4f}")
    print_metric("LCS Length", f"{adherence.lcs_length} / {max(adherence.plan_length, adherence.actual_length)}")
    print_metric("Matched Steps", ", ".join(adherence.matched_steps))
    print_metric("Missed Steps", ", ".join(adherence.missed_steps) or "(none)", Colors.YELLOW)
    print_metric("Extra Steps", ", ".join(adherence.extra_steps) or "(none)", Colors.YELLOW)

    # ─── LLM-as-a-Judge ──────────────────────────────────────────────
    print_subheader("LLM-as-a-Judge Evaluation (Offline Mode)")

    objective = "Find a top-rated restaurant and make a dinner reservation for 2 people."
    plan_json = json.dumps(plan)

    quality = judge.evaluate_plan_quality(objective, plan_json)
    print_metric("Plan Quality Score", f"{quality.score:.4f}")
    for dim, score in quality.sub_scores.items():
        print_metric(f"  └─ {dim.capitalize()}", f"{score:.4f}", Colors.BLUE)
    print(f"  {Colors.DIM}│  Reasoning: {quality.reasoning}{Colors.RESET}")

    trace_json = json.dumps(actual_steps)
    adherence_judge = judge.evaluate_plan_adherence(plan_json, trace_json)
    print()
    print_metric("Plan Adherence Score (Judge)", f"{adherence_judge.score:.4f}")
    for dim, score in adherence_judge.sub_scores.items():
        print_metric(f"  └─ {dim.replace('_', ' ').capitalize()}", f"{score:.4f}", Colors.BLUE)
    print(f"  {Colors.DIM}│  Reasoning: {adherence_judge.reasoning}{Colors.RESET}")

    # ─── Elo Rating ──────────────────────────────────────────────────
    print_subheader("Elo Performance Rating")

    elo = elo_rating(wins=7, losses=2, draws=1, k_factor=32.0)
    print_metric("Final Elo Rating", f"{elo['final_rating']:.2f}")
    print_metric("Expected Score", f"{elo['expected_score']:.4f}")
    print_metric("Actual Score", f"{elo['actual_score']:.4f}")
    print_metric("Total Matches", elo["total_matches"])

    # ─── Composite Summary ───────────────────────────────────────────
    print_subheader("📊 Composite Evaluation Summary")

    composite = suite.full_evaluation(
        agent_trace=agent_trace,
        ground_truth_tools=ground_truth,
        plan=plan,
        actual_steps=actual_steps,
    )

    print()
    print(f"  {Colors.BOLD}┌─────────────────────────────────────┬───────────┐{Colors.RESET}")
    print(f"  {Colors.BOLD}│ Metric                              │   Score   │{Colors.RESET}")
    print(f"  {Colors.BOLD}├─────────────────────────────────────┼───────────┤{Colors.RESET}")

    metrics = [
        ("Tool Correctness", composite["tool_evaluation"]["tool_correctness"]),
        ("Tool Efficiency", composite["tool_evaluation"]["tool_efficiency"]),
        ("Redundancy Rate", composite["tool_evaluation"]["redundancy_rate"]),
        ("Plan Adherence (LCS)", composite["plan_evaluation"]["adherence_score"]),
        ("Plan Quality (Judge)", quality.score),
        ("Plan Adherence (Judge)", adherence_judge.score),
        ("Elo Rating", elo["final_rating"] / 2000),
    ]

    for name, score in metrics:
        bar_len = int(score * 20)
        bar = f"{Colors.GREEN}{'█' * bar_len}{Colors.DIM}{'░' * (20 - bar_len)}{Colors.RESET}"
        print(f"  │ {name:<35} │ {score:>7.4f}   │  {bar}")

    print(f"  {Colors.BOLD}├─────────────────────────────────────┼───────────┤{Colors.RESET}")
    print(f"  {Colors.BOLD}│ {'COMPOSITE SCORE':<35} │ {composite['composite_score']:>7.4f}   │{Colors.RESET}")
    print(f"  {Colors.BOLD}└─────────────────────────────────────┴───────────┘{Colors.RESET}")

    print(f"\n  {Colors.GREEN}✓ Agent-Bench demo complete.{Colors.RESET}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run all demonstration pipelines."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("  ╔═══════════════════════════════════════════════════════════════╗")
    print("  ║                                                             ║")
    print("  ║   ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗              ║")
    print("  ║   ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗             ║")
    print("  ║   ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║             ║")
    print("  ║   ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║             ║")
    print("  ║   ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝             ║")
    print("  ║   ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝             ║")
    print("  ║                                                             ║")
    print("  ║   F U S I O N   B E N C H   v0.1.0                         ║")
    print("  ║   Scientific Framework for Neural Model Experimentation     ║")
    print("  ║                                                             ║")
    print("  ╚═══════════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}")

    try:
        demo_model_merging()
        demo_swi_reasoning()
        demo_agent_bench()
    except Exception as e:
        print(f"\n  {Colors.RED}✗ Error: {e}{Colors.RESET}")
        raise

    print(f"\n{Colors.BOLD}{Colors.GREEN}")
    print("  ═══════════════════════════════════════════════════════════════")
    print("  ✓  All demonstrations completed successfully.")
    print("  ═══════════════════════════════════════════════════════════════")
    print(f"{Colors.RESET}")


if __name__ == "__main__":
    main()
