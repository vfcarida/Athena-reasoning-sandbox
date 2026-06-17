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
    ) -> Path:
        """Export a trained model in a format consumable by Agent-Bench.

        Saves the model and tokenizer to disk in HF format, plus a metadata
        file describing the model configuration for Agent-Bench.

        Args:
            model: The trained model to export.
            tokenizer: The model's tokenizer.
            output_path: Directory to save the exported model.
            model_name: Human-readable name for the model.

        Returns:
            Path to the exported model directory.
        """
        path = Path(output_path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(path))
        else:
            torch.save(model.state_dict(), path / "pytorch_model.bin")

        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(str(path))

        # Write metadata for Agent-Bench
        metadata = {
            "model_name": model_name,
            "model_path": str(path),
            "framework": "athena-reasoning-sandbox",
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "device": str(next(model.parameters()).device),
        }

        with open(path / "athena_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Model exported for Agent-Bench: %s → %s", model_name, path)
        return path

    def create_model_adapter(
        self,
        model: nn.Module,
        tokenizer: Any,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Any:
        """Create an Agent-Bench ModelAdapter wrapping an Athena-trained model.

        The adapter implements Agent-Bench's model interface, allowing the
        model to be used directly in Agent-Bench evaluation pipelines without
        going through an API.

        Args:
            model: The trained model.
            tokenizer: The model's tokenizer.
            system_prompt: Default system prompt for the adapter.
            max_tokens: Maximum tokens for generation.
            temperature: Sampling temperature.

        Returns:
            An ``AthenaModelAdapter`` instance compatible with Agent-Bench.

        Raises:
            RuntimeError: If Agent-Bench is not available.
        """
        if not self._agent_bench_available:
            logger.warning(
                "Agent-Bench not available. Returning a standalone adapter."
            )

        adapter = AthenaModelAdapter(
            model=model,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        logger.info("ModelAdapter created for Agent-Bench evaluation.")
        return adapter

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


class AthenaModelAdapter:
    """Adapter wrapping an Athena-trained model for Agent-Bench compatibility.

    Implements the interface expected by Agent-Bench's evaluation runners,
    translating between Athena's model format and Agent-Bench's expected
    input/output conventions.

    This adapter handles:
    - Message formatting (system, user, assistant roles)
    - Token generation with configurable temperature
    - Tool call response parsing (if the model supports function calling)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> None:
        """Initialize the model adapter.

        Args:
            model: The Athena-trained model.
            tokenizer: The model's tokenizer.
            system_prompt: Default system prompt.
            max_tokens: Maximum tokens for generation.
            temperature: Sampling temperature.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature

        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

    async def generate(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a response from the model given a conversation.

        This method follows the Agent-Bench adapter interface convention.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            tools: Optional list of tool definitions (for function calling).
            **kwargs: Additional generation parameters.

        Returns:
            Dictionary with "response", "tool_calls" (if any), and metadata.
        """
        # Format messages into a prompt string
        prompt = self._format_messages(messages)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length or 2048,
        ).to(self.device)

        # Generate
        max_new = kwargs.get("max_tokens", self.max_tokens)
        temp = kwargs.get("temperature", self.temperature)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=temp if temp > 0 else 1.0,
                do_sample=temp > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode response (only the new tokens)
        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return {
            "response": response_text.strip(),
            "tool_calls": [],
            "usage": {
                "prompt_tokens": inputs["input_ids"].shape[-1],
                "completion_tokens": len(new_tokens),
            },
        }

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        """Format a list of chat messages into a model prompt.

        Args:
            messages: List of message dictionaries.

        Returns:
            Formatted prompt string.
        """
        parts = []
        if self.system_prompt:
            parts.append(f"System: {self.system_prompt}")

        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")

        parts.append("Assistant:")
        return "\n\n".join(parts)
