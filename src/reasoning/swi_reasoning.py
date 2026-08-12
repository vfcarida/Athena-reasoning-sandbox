"""SwiReasoning Engine — Entropy-Guided Adaptive Inference.

This module implements the SwiReasoning mechanism: a token-by-token autoregressive
generation loop that monitors the Shannon entropy of output logits in real-time
and dynamically switches between two inference modes:

- **Explicit Mode** (default): Standard token generation visible to the user.
- **Latent (Thinking) Mode**: Triggered when entropy exceeds a confidence threshold,
  indicating model uncertainty. The engine injects ``<think>`` control tokens and
  switches to a more exploratory sampling strategy. When confidence is restored
  (entropy drops below threshold), the engine injects ``</think>`` and resumes
  explicit generation.

An overthinking prevention system enforces hard limits on:
- Maximum number of mode switches (``max_switches``)
- Maximum consecutive tokens in thinking mode (``max_thinking_tokens``)

This prevents infinite reasoning loops and excessive compute waste.

Algorithmic Complexity:
    - Logit Entropy Calculation: O(V) where V is vocabulary size.
    - Autoregressive Step: O(L * V) where L is sequence length and V is vocabulary size.

References:
    - Shannon, C. E. (1948). A Mathematical Theory of Communication.
    - DeepSeek-R1 and QwQ reasoning paradigms for thinking token injection.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from src.reasoning.schemas import ActionType, AgentPlan, ReasoningStep

logger = logging.getLogger(__name__)


class InferenceMode(Enum):
    """Operating modes for the SwiReasoning engine."""
    EXPLICIT = auto()  # Standard visible generation
    LATENT = auto()    # Internal thinking / reasoning mode


@dataclass
class GenerationState:
    """Tracks the internal state of the SwiReasoning generation loop.

    Attributes:
        mode: Current inference mode (EXPLICIT or LATENT).
        switch_count: Number of mode transitions performed so far.
        thinking_token_count: Consecutive tokens generated in thinking mode.
        total_thinking_tokens: Total tokens generated in thinking mode across
            all thinking phases.
        total_tokens: Total tokens generated (both modes).
        entropy_history: List of entropy values computed at each generation step.
    """
    mode: InferenceMode = InferenceMode.EXPLICIT
    switch_count: int = 0
    thinking_token_count: int = 0
    total_thinking_tokens: int = 0
    total_tokens: int = 0
    entropy_history: List[float] = field(default_factory=list)


class SwiReasoningEngine:
    """Entropy-guided adaptive inference engine with dynamic mode switching.

    The engine wraps a Hugging Face ``PreTrainedModel`` and ``PreTrainedTokenizer``
    to provide a custom autoregressive generation loop. At each decoding step, the
    Shannon entropy of the next-token probability distribution is computed. If
    entropy exceeds a configurable threshold, the engine enters "thinking" mode
    to allow the model to reason internally before producing visible output.
    """

    THINK_START_TOKEN = "<think>"
    THINK_END_TOKEN = "</think>"

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        entropy_threshold: float = 1.5,
        max_switches: int = 2,
        max_thinking_tokens: int = 64,
        thinking_temperature: float = 1.2,
        explicit_temperature: float = 0.7,
    ) -> None:
        """Initialize the SwiReasoning engine.

        Args:
            model: A Hugging Face ``PreTrainedModel`` with a language modeling head.
            tokenizer: Corresponding ``PreTrainedTokenizer`` for the model.
            entropy_threshold: Shannon entropy threshold (in bits) above which the
                engine considers the model "uncertain" and switches to thinking mode.
            max_switches: Maximum number of EXPLICIT->LATENT mode transitions allowed.
            max_thinking_tokens: Maximum consecutive tokens allowed in thinking mode.
            thinking_temperature: Sampling temperature during thinking mode.
            explicit_temperature: Sampling temperature during explicit mode.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.entropy_threshold = entropy_threshold
        self.max_switches = max_switches
        self.max_thinking_tokens = max_thinking_tokens
        self.thinking_temperature = thinking_temperature
        self.explicit_temperature = explicit_temperature

        try:
            self.device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            self.device = torch.device("cpu")

        logger.info(
            "SwiReasoningEngine initialized: entropy_threshold=%.2f, "
            "max_switches=%d, max_thinking_tokens=%d, device=%s",
            entropy_threshold, max_switches, max_thinking_tokens, self.device,
        )

    @staticmethod
    def calculate_entropy(logits: torch.Tensor, eps: float = 1e-12) -> float:
        """Compute Shannon entropy from raw logits.

        Converts logits to a probability distribution via softmax, then computes:

            H(X) = -sum_i P(x_i) * log_2(P(x_i))

        Args:
            logits: Raw unnormalized logit tensor of shape ``(vocab_size,)``
                or ``(1, vocab_size)``.
            eps: Small epsilon for numerical stability.

        Returns:
            Shannon entropy in bits (float). Returns 0.0 for degenerate inputs.
        """
        if logits.dim() > 1:
            logits = logits.squeeze(0)

        logits = logits.to(torch.float64)
        probs = F.softmax(logits, dim=-1)
        probs = torch.clamp(probs, min=eps)

        log_probs = torch.log2(probs)
        entropy = -torch.sum(probs * log_probs).item()

        if not (isinstance(entropy, float) and entropy == entropy and entropy != float("inf")):
            return 0.0

        return max(entropy, 0.0)

    def generate_with_switch_thinking(
        self,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> GenerationResult:
        """Run synchronous autoregressive generation with entropy-guided mode switching.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            A GenerationResult containing output text and generation stats.
        """
        state = GenerationState()

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.device)

        generated_tokens: List[str] = []
        all_tokens: List[str] = []
        current_ids = input_ids

        logger.info("Starting SwiReasoning generation: max_new_tokens=%d", max_new_tokens)

        for step in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(current_ids)
                logits = outputs.logits[:, -1, :]

            entropy = self.calculate_entropy(logits)
            state.entropy_history.append(entropy)

            if state.mode == InferenceMode.EXPLICIT:
                if (
                    entropy > self.entropy_threshold
                    and state.switch_count < self.max_switches
                ):
                    state.mode = InferenceMode.LATENT
                    state.switch_count += 1
                    state.thinking_token_count = 0
                    all_tokens.append(self.THINK_START_TOKEN)

            elif state.mode == InferenceMode.LATENT:
                should_exit = (
                    entropy < self.entropy_threshold
                    or state.thinking_token_count >= self.max_thinking_tokens
                )
                if should_exit:
                    state.mode = InferenceMode.EXPLICIT
                    all_tokens.append(self.THINK_END_TOKEN)
                    state.thinking_token_count = 0

            temperature = (
                self.thinking_temperature
                if state.mode == InferenceMode.LATENT
                else self.explicit_temperature
            )

            scaled_logits = logits / temperature
            probs = F.softmax(scaled_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            token_text = self.tokenizer.decode(
                next_token_id[0], skip_special_tokens=False
            )

            all_tokens.append(token_text)
            state.total_tokens += 1

            if state.mode == InferenceMode.LATENT:
                state.thinking_token_count += 1
                state.total_thinking_tokens += 1
            else:
                generated_tokens.append(token_text)

            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break

            current_ids = torch.cat([current_ids, next_token_id], dim=-1)

        output_text = "".join(all_tokens)
        visible_text = "".join(generated_tokens)

        return GenerationResult(
            output_text=output_text,
            visible_text=visible_text,
            state=state,
            prompt=prompt,
        )

    async def generate_with_switch_thinking_async(
        self,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> GenerationResult:
        """Asynchronous wrapper for non-blocking LLM inference.

        Executes generation in an asyncio threadpool executor to avoid blocking
        the main async event loop.

        Args:
            prompt: Input prompt text.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            GenerationResult object.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.generate_with_switch_thinking,
            prompt,
            max_new_tokens,
        )


@dataclass
class GenerationResult:
    """Container for SwiReasoning generation output.

    Attributes:
        output_text: Full generated text including ``<think>``/``</think>`` markers.
        visible_text: Only the tokens generated in explicit (visible) mode.
        state: Final GenerationState with counters and entropy history.
        prompt: The original input prompt.
    """
    output_text: str
    visible_text: str
    state: GenerationState
    prompt: str

    def summary(self) -> Dict[str, Union[int, float]]:
        """Return a summary dictionary of the generation run.

        Returns:
            Dictionary with key statistics.
        """
        entropy_hist = self.state.entropy_history
        avg_entropy = sum(entropy_hist) / len(entropy_hist) if entropy_hist else 0.0
        max_entropy = max(entropy_hist) if entropy_hist else 0.0
        min_entropy = min(entropy_hist) if entropy_hist else 0.0

        total = self.state.total_tokens
        thinking = self.state.total_thinking_tokens
        ot_ratio = thinking / total if total > 0 else 0.0

        return {
            "total_tokens": total,
            "visible_tokens": total - thinking,
            "thinking_tokens": thinking,
            "switch_count": self.state.switch_count,
            "overthinking_ratio": round(ot_ratio, 4),
            "entropy_avg": round(avg_entropy, 4),
            "entropy_max": round(max_entropy, 4),
            "entropy_min": round(min_entropy, 4),
        }

    def to_agent_plan(self) -> AgentPlan:
        """Extract a structured AgentPlan object from generation output.

        Returns:
            Pydantic AgentPlan object representing the Plan-then-Execute trajectory.
        """
        steps = [
            ReasoningStep(
                step_number=1,
                rationale="Execute primary generated task based on explicit inference.",
                action_type=ActionType.FINAL_ANSWER,
                tool_name="final_answer",
                tool_args={"text": self.visible_text},
            )
        ]
        return AgentPlan(
            plan_id=str(uuid.uuid4()),
            task_goal=self.prompt,
            thinking_process=self.output_text if "<think>" in self.output_text else None,
            steps=steps,
            estimated_complexity=1,
        )


class SwiReasoningSimulator:
    """Lightweight simulator for SwiReasoning without requiring a real GPU model."""

    def __init__(
        self,
        vocab_size: int = 32000,
        entropy_threshold: float = 2.0,
        max_switches: int = 3,
        max_thinking_tokens: int = 32,
        seed: Optional[int] = 42,
    ) -> None:
        """Initialize the simulator."""
        self.vocab_size = vocab_size
        self.entropy_threshold = entropy_threshold
        self.max_switches = max_switches
        self.max_thinking_tokens = max_thinking_tokens

        if seed is not None:
            torch.manual_seed(seed)

    def _generate_synthetic_logits(self, step: int, num_steps: int) -> torch.Tensor:
        """Generate synthetic logits with varying entropy levels."""
        import math

        phase = step / max(num_steps - 1, 1)
        uncertainty = math.sin(phase * math.pi) * 2.5 + 0.5
        concentration = max(0.1, 5.0 - uncertainty * 2.0)
        logits = torch.randn(self.vocab_size) * concentration

        if concentration > 2.0:
            top_idx = torch.randint(0, self.vocab_size, (1,)).item()
            logits[top_idx] += concentration * 3.0

        return logits

    def simulate(
        self,
        prompt: str,
        num_steps: int = 30,
    ) -> GenerationResult:
        """Run a simulated SwiReasoning generation."""
        state = GenerationState()
        all_tokens: List[str] = []
        visible_tokens: List[str] = []

        sample_words = [
            "the", "model", "processes", "input", "data",
            "with", "neural", "network", "layers", "to",
            "generate", "accurate", "predictions", "using",
            "advanced", "reasoning", "capabilities",
        ]

        for step in range(num_steps):
            logits = self._generate_synthetic_logits(step, num_steps)
            entropy = SwiReasoningEngine.calculate_entropy(logits)
            state.entropy_history.append(entropy)

            if state.mode == InferenceMode.EXPLICIT:
                if (
                    entropy > self.entropy_threshold
                    and state.switch_count < self.max_switches
                ):
                    state.mode = InferenceMode.LATENT
                    state.switch_count += 1
                    state.thinking_token_count = 0
                    all_tokens.append(SwiReasoningEngine.THINK_START_TOKEN)

            elif state.mode == InferenceMode.LATENT:
                should_exit = (
                    entropy < self.entropy_threshold
                    or state.thinking_token_count >= self.max_thinking_tokens
                )
                if should_exit:
                    state.mode = InferenceMode.EXPLICIT
                    all_tokens.append(SwiReasoningEngine.THINK_END_TOKEN)
                    state.thinking_token_count = 0

            word_idx = step % len(sample_words)
            token = sample_words[word_idx]
            all_tokens.append(f" {token}")
            state.total_tokens += 1

            if state.mode == InferenceMode.LATENT:
                state.thinking_token_count += 1
                state.total_thinking_tokens += 1
            else:
                visible_tokens.append(f" {token}")

        if state.mode == InferenceMode.LATENT:
            all_tokens.append(SwiReasoningEngine.THINK_END_TOKEN)

        return GenerationResult(
            output_text="".join(all_tokens),
            visible_text="".join(visible_tokens),
            state=state,
            prompt=prompt,
        )

    async def simulate_async(
        self,
        prompt: str,
        num_steps: int = 30,
    ) -> GenerationResult:
        """Asynchronously run simulated SwiReasoning generation."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.simulate, prompt, num_steps)
