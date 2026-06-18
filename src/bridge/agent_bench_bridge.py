"""Agent-Bench Bridge — Integration adapter for the Agent-Bench evaluation framework.

This module provides the connection between Athena Reasoning Sandbox (model
development) and Agent-Bench (model evaluation). It enables models trained
in this sandbox to be evaluated using Agent-Bench's comprehensive suite of
deterministic and semantic graders.

Integration modes:
    - **Direct Import**: Import Agent-Bench as a Python package (requires
      ``pip install -e ../Agent-Bench``).
    - **CLI Invocation**: Call Agent-Bench's ``bench`` CLI for evaluation runs.
    - **Adapter Export**: Package a trained model as an Agent-Bench ModelAdapter.

The bridge abstracts Agent-Bench's internal API so that experimentation code
in Athena can trigger evaluations without knowledge of Agent-Bench internals.

References:
    - Agent-Bench repo: https://github.com/vfcarida/Agent-Bench
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class BridgeConfig:
    """Configuration for the Agent-Bench bridge.

    Attributes:
        agent_bench_path: Path to the local Agent-Bench repository.
        config_dir: Path to Agent-Bench configuration directory.
        output_dir: Directory for storing evaluation results.
        default_suite: Default evaluation suite ID.
    """
    agent_bench_path: str = "../Agent-Bench"
    config_dir: str = "../Agent-Bench/configs"
    output_dir: str = "./evaluation_results"
    default_suite: str = "pix_basic_v1"


class AgentBenchBridge:
    """Bridge between Athena model experiments and Agent-Bench evaluation.

    Provides methods to export trained models in Agent-Bench compatible format,
    invoke evaluations via the Agent-Bench framework, and compare results
    across multiple experimental configurations.

    Example:
        >>> bridge = AgentBenchBridge()
        >>> adapter = bridge.create_model_adapter(model, tokenizer)
        >>> results = bridge.run_benchmark("pix_basic_v1")
    """

    def __init__(self, config: Optional[BridgeConfig] = None) -> None:
        """Initialize the Agent-Bench bridge.

        Args:
            config: Bridge configuration. Uses defaults if None.
        """
        self.config = config or BridgeConfig()
        self._agent_bench_available = self._check_agent_bench()

        logger.info(
            "AgentBenchBridge initialized: agent_bench=%s, available=%s",
            self.config.agent_bench_path, self._agent_bench_available,
        )

    def _check_agent_bench(self) -> bool:
        """Check if Agent-Bench is available as a Python package.

        Returns:
            True if Agent-Bench can be imported, False otherwise.
        """
        try:
            import agent_bench  # noqa: F401
            return True
        except ImportError:
            bench_path = Path(self.config.agent_bench_path)
            if bench_path.exists():
                # Try adding to path
                src_path = str(bench_path / "src")
                if src_path not in sys.path:
                    sys.path.insert(0, src_path)
                try:
                    import agent_bench  # noqa: F401
                    return True
                except ImportError:
                    pass

            logger.warning(
                "Agent-Bench not available. Install with: "
                "pip install -e %s", self.config.agent_bench_path,
            )
            return False

    @property
    def is_available(self) -> bool:
        """Whether Agent-Bench is available for evaluation."""
        return self._agent_bench_available

    def export_for_evaluation(
        self,
        model: nn.Module,
        tokenizer: Any,
        output_path: str | Path,
        model_name: str = "athena-experiment",
        provider: str = "huggingface",
        prompt_template: str = "chatml",
        adapter_path: Optional[str] = None,
        adapter_name: Optional[str] = None,
        has_thinking_tokens: bool = False,
        lineage: Optional[dict[str, Any]] = None,
    ) -> Path:
        """Export a trained model in a format consumable by Agent-Bench.

        Saves the model and tokenizer to disk in HF format, and writes an
        `athena_experiment.yaml` configuration file compatible with Agent-Bench.

        Args:
            model: The trained model to export.
            tokenizer: The model's tokenizer.
            output_path: Directory to save the exported model.
            model_name: Human-readable name (model_id in Agent-Bench).
            provider: Inference provider (e.g., huggingface, vllm).
            prompt_template: Prompt formatting template (e.g., chatml).
            adapter_path: Path to PEFT adapter, if applicable.
            adapter_name: Name of the PEFT adapter.
            has_thinking_tokens: Whether to enable Agent-Bench thinking_parser.
            lineage: Metadata about parent models and merge operations.

        Returns:
            Path to the exported model directory.
        """
        import yaml
        path = Path(output_path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(path))
        else:
            torch.save(model.state_dict(), path / "pytorch_model.bin")

        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(str(path))

        # Generate Agent-Bench compatible YAML config
        model_config = {
            "model_id": model_name,
            "provider": provider,
            "model_path": str(path),
            "prompt_template": prompt_template,
            "device": "auto",
            "parameters": {
                "temperature": 0.0,
                "max_tokens": 4096
            }
        }

        if provider == "vllm":
            model_config["tensor_parallel_size"] = 1
            model_config["gpu_memory_utilization"] = 0.9
        else:
            model_config["torch_dtype"] = "bfloat16"

        if adapter_path:
            model_config["adapter_path"] = adapter_path
            if adapter_name:
                model_config["adapter_name"] = adapter_name

        if has_thinking_tokens:
            model_config["has_thinking_tokens"] = True

        if lineage:
            model_config["lineage"] = lineage

        config_doc = {"models": [model_config]}

        yaml_path = path / "athena_experiment.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config_doc, f, default_flow_style=False, sort_keys=False)

        # Legacy metadata file for debugging
        metadata = {
            "model_name": model_name,
            "model_path": str(path),
            "framework": "athena-reasoning-sandbox",
            "num_parameters": sum(p.numel() for p in model.parameters()),
        }
        with open(path / "athena_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Model exported for Agent-Bench: %s → %s", model_name, path)
        return path

    def create_model_adapter(
        self,
        model_path: str,
        provider: str = "huggingface",
        adapter_path: Optional[str] = None,
    ) -> Any:
        """Create a native Agent-Bench ModelAdapter wrapping an Athena-trained model.

        Instead of duplicating adapter logic, this delegates to Agent-Bench's
        native implementations (e.g., HuggingFacePipelineAdapter, PEFTModelAdapter).

        Args:
            model_path: Local path to the exported model.
            provider: The Agent-Bench provider type ("huggingface", "vllm").
            adapter_path: Path to LoRA weights (triggers PEFTAdapter if set).

        Returns:
            A native Agent-Bench adapter instance.

        Raises:
            RuntimeError: If Agent-Bench is not available.
        """
        if not self._agent_bench_available:
            raise RuntimeError(
                "Agent-Bench is not installed. Native adapters cannot be created. "
                "Install with `pip install -e ../Agent-Bench`."
            )

        from agent_bench.models import get_huggingface_adapter, get_vllm_adapter, get_peft_adapter

        if provider == "vllm":
            adapter_cls = get_vllm_adapter()
            return adapter_cls(model_path=model_path, tensor_parallel_size=1)
        
        if adapter_path:
            adapter_cls = get_peft_adapter()
            return adapter_cls(model_path=model_path, adapter_path=adapter_path)
            
        adapter_cls = get_huggingface_adapter()
        return adapter_cls(model_path=model_path, device="auto", torch_dtype="auto")

    def run_benchmark(
        self,
        suite_id: Optional[str] = None,
        config_dir: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run an Agent-Bench evaluation suite via CLI.

        Invokes the Agent-Bench ``bench`` CLI tool to execute a benchmark suite
        and returns the results.

        Args:
            suite_id: Evaluation suite identifier. Defaults to config.default_suite.
            config_dir: Path to Agent-Bench configs. Defaults to config.config_dir.

        Returns:
            Dictionary with evaluation results.

        Raises:
            FileNotFoundError: If Agent-Bench CLI is not found.
        """
        suite = suite_id or self.config.default_suite
        configs = config_dir or self.config.config_dir

        cmd = [
            sys.executable, "-m", "agent_bench.cli.main",
            "--config-dir", configs,
            "run-suite", suite,
        ]

        logger.info("Running Agent-Bench: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "suite_id": suite,
                "returncode": result.returncode,
            }

        except FileNotFoundError:
            logger.error("Agent-Bench CLI not found. Install with: pip install -e %s",
                        self.config.agent_bench_path)
            return {"success": False, "error": "Agent-Bench CLI not found"}

        except subprocess.TimeoutExpired:
            logger.error("Agent-Bench evaluation timed out.")
            return {"success": False, "error": "Evaluation timed out"}

    def compare_experiments(
        self,
        experiment_results: list[dict[str, Any]],
        metric_keys: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Compare evaluation results across multiple model experiments.

        Generates a comparative summary of metrics from different Athena
        experiments evaluated by Agent-Bench.

        Args:
            experiment_results: List of result dictionaries from evaluations.
            metric_keys: Specific metric keys to compare. If None, compares
                all available metrics.

        Returns:
            Comparison summary with rankings and statistical analysis.
        """
        if not experiment_results:
            return {"error": "No experiment results to compare"}

        comparison = {
            "num_experiments": len(experiment_results),
            "experiments": [],
        }

        for i, result in enumerate(experiment_results):
            exp_summary = {
                "experiment_id": i,
                "suite_id": result.get("suite_id", "unknown"),
                "success": result.get("success", False),
            }
            comparison["experiments"].append(exp_summary)

        logger.info("Compared %d experiments.", len(experiment_results))
        return comparison

    def get_agent_bench_metrics(self) -> dict[str, str]:
        """Return the list of metrics available in Agent-Bench.

        Provides a reference of what Agent-Bench measures, corresponding
        to the evaluation metrics described in the strategic framework:
        - Plan Quality, Plan Adherence
        - Tool Correctness, Tool Efficiency
        - Task Completion
        - Context Precision/Recall/Relevancy

        Returns:
            Dictionary mapping metric names to descriptions.
        """
        return {
            "functional_score": "Weighted functional correctness (state + tool + rubric grading)",
            "risk_score": "Safety and policy compliance (refusal accuracy, forbidden actions)",
            "cost_score": "Token usage and API invocation cost efficiency",
            "latency_score": "End-to-end response time of the agent system",
            "reliability_score": "Pass@k reliability across repeated runs",
            "tool_correctness": "Precision of tool calls against expected sequence",
            "tool_efficiency": "Ratio of optimal to actual tool invocations",
            "plan_quality": "Semantic coherence and feasibility of agent plans",
            "plan_adherence": "Sequential alignment between plan and execution",
            "task_completion": "Binary/graded success on the primary objective",
            "context_precision": "Accuracy of retrieved context for RAG tasks",
            "context_recall": "Coverage of relevant information in retrieval",
            "context_relevancy": "Signal-to-noise ratio in provided context",
        }



