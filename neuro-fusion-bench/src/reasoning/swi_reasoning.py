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

References:
    - Shannon, C. E. (1948). A Mathematical Theory of Communication.
    - DeepSeek-R1 and QwQ reasoning paradigms for thinking token injection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import torch
import torch.nn.functional as F

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
    entropy_history: list[float] = field(default_factory=list)


class SwiReasoningEngine:
    """Entropy-guided adaptive inference engine with dynamic mode switching.

    The engine wraps a Hugging Face ``PreTrainedModel`` and ``PreTrainedTokenizer``
    to provide a custom autoregressive generation loop. At each decoding step, the
    Shannon entropy of the next-token probability distribution is computed. If
    entropy exceeds a configurable threshold, the engine enters "thinking" mode
    to allow the model to reason internally before producing visible output.

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> engine = SwiReasoningEngine(model, tokenizer, entropy_threshold=2.0)
        >>> result = engine.generate_with_switch_thinking("Explain quantum computing:")
        >>> print(result.output_text)
    """

    # Control tokens injected into the generation stream
    THINK_START_TOKEN = "<think>"
    THINK_END_TOKEN = "</think>"

    def __init__(
        self,
        model: object,
        tokenizer: object,
        entropy_threshold: float = 1.5,
        max_switches: int = 2,
        max_thinking_tokens: int = 64,
        thinking_temperature: float = 1.2,
        explicit_temperature: float = 0.7,
    ) -> None:
        """Initialize the SwiReasoning engine.

        Args:
            model: A Hugging Face ``PreTrainedModel`` with a language modeling head
                (e.g., ``AutoModelForCausalLM``).
            tokenizer: Corresponding ``PreTrainedTokenizer`` for the model.
            entropy_threshold: Shannon entropy threshold (in bits) above which the
                engine considers the model "uncertain" and switches to thinking mode.
                Typical values: 1.0–3.0 depending on vocabulary size.
            max_switches: Maximum number of EXPLICIT→LATENT mode transitions allowed
                per generation call. Prevents infinite oscillation.
            max_thinking_tokens: Maximum consecutive tokens allowed in thinking mode
                before forced exit. Prevents runaway internal reasoning.
            thinking_temperature: Sampling temperature during thinking mode. Higher
                values encourage exploration. Default: 1.2.
            explicit_temperature: Sampling temperature during explicit mode. Lower
                values favor greedy/confident output. Default: 0.7.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.entropy_threshold = entropy_threshold
        self.max_switches = max_switches
        self.max_thinking_tokens = max_thinking_tokens
        self.thinking_temperature = thinking_temperature
        self.explicit_temperature = explicit_temperature

        # Determine device from model parameters
        try:
            self.device = next(model.parameters()).device  # type: ignore[union-attr]
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

            H(X) = −Σᵢ P(xᵢ) · log₂(P(xᵢ))

        Safe against numerical issues: clamps near-zero probabilities to avoid
        log(0) and handles NaN/Inf results gracefully.

        Args:
            logits: Raw unnormalized logit tensor of shape ``(vocab_size,)``
                or ``(1, vocab_size)``.
            eps: Small epsilon for numerical stability.

        Returns:
            Shannon entropy in bits (float). Returns 0.0 for degenerate inputs.
        """
        # Ensure 1-D
        if logits.dim() > 1:
            logits = logits.squeeze(0)

        # Convert to float64 for numerical precision
        logits = logits.to(torch.float64)

        # Softmax to get probabilities
        probs = F.softmax(logits, dim=-1)

        # Clamp to avoid log(0)
        probs = torch.clamp(probs, min=eps)

        # Shannon entropy in bits (base 2)
        log_probs = torch.log2(probs)
        entropy = -torch.sum(probs * log_probs).item()

        # Guard against NaN/Inf
        if not (isinstance(entropy, float) and entropy == entropy and entropy != float("inf")):
            return 0.0

        return max(entropy, 0.0)

    def generate_with_switch_thinking(
        self,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> "GenerationResult":
        """Run autoregressive generation with entropy-guided mode switching.

        Performs token-by-token decoding. At each step:
        1. Forward pass to get next-token logits.
        2. Compute Shannon entropy of the logit distribution.
        3. If in EXPLICIT mode and entropy > threshold → switch to LATENT
           (inject ``<think>``).
        4. If in LATENT mode and entropy < threshold → switch to EXPLICIT
           (inject ``</think>``).
        5. Enforce overthinking limits (max_switches, max_thinking_tokens).
        6. Sample next token using mode-appropriate temperature.

        Args:
            prompt: Input text prompt to continue generating from.
            max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            A :class:`GenerationResult` containing the full output text,
            visible-only text, generation state, and per-token entropy trace.
        """
        state = GenerationState()

        # Tokenize the input prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")  # type: ignore[union-attr]
        input_ids = input_ids.to(self.device)

        generated_tokens: list[str] = []
        all_tokens: list[str] = []
        current_ids = input_ids

        logger.info("Starting SwiReasoning generation: max_new_tokens=%d", max_new_tokens)

        for step in range(max_new_tokens):
            # Forward pass (no gradient computation needed for inference)
            with torch.no_grad():
                outputs = self.model(current_ids)  # type: ignore[operator]
                logits = outputs.logits[:, -1, :]  # Shape: (1, vocab_size)

            # Compute entropy of the next-token distribution
            entropy = self.calculate_entropy(logits)
            state.entropy_history.append(entropy)

            # ─────────────────────────────────────────────────────────────
            # State Machine: Mode Transition Logic
            # ─────────────────────────────────────────────────────────────

            if state.mode == InferenceMode.EXPLICIT:
                # Check if we should enter thinking mode
                if (
                    entropy > self.entropy_threshold
                    and state.switch_count < self.max_switches
                ):
                    state.mode = InferenceMode.LATENT
                    state.switch_count += 1
                    state.thinking_token_count = 0
                    all_tokens.append(self.THINK_START_TOKEN)
                    logger.debug(
                        "Step %d: EXPLICIT → LATENT (entropy=%.4f > %.4f, switch #%d)",
                        step, entropy, self.entropy_threshold, state.switch_count,
                    )

            elif state.mode == InferenceMode.LATENT:
                # Check if we should exit thinking mode
                should_exit = (
                    entropy < self.entropy_threshold
                    or state.thinking_token_count >= self.max_thinking_tokens
                )

                if should_exit:
                    forced = state.thinking_token_count >= self.max_thinking_tokens
                    state.mode = InferenceMode.EXPLICIT
                    all_tokens.append(self.THINK_END_TOKEN)
                    logger.debug(
                        "Step %d: LATENT → EXPLICIT (%s, entropy=%.4f, "
                        "thinking_tokens=%d)",
                        step,
                        "FORCED EXIT" if forced else "entropy dropped",
                        entropy,
                        state.thinking_token_count,
                    )
                    state.thinking_token_count = 0

            # ─────────────────────────────────────────────────────────────
            # Token Sampling with Mode-Dependent Temperature
            # ─────────────────────────────────────────────────────────────

            temperature = (
                self.thinking_temperature
                if state.mode == InferenceMode.LATENT
                else self.explicit_temperature
            )

            scaled_logits = logits / temperature
            probs = F.softmax(scaled_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            # Decode the sampled token
            token_text = self.tokenizer.decode(  # type: ignore[union-attr]
                next_token_id[0], skip_special_tokens=False
            )

            # Track tokens
            all_tokens.append(token_text)
            state.total_tokens += 1

            if state.mode == InferenceMode.LATENT:
                state.thinking_token_count += 1
                state.total_thinking_tokens += 1
            else:
                generated_tokens.append(token_text)

            # Check for EOS
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                logger.debug("Step %d: EOS token encountered.", step)
                break

            # Append to sequence for next iteration
            current_ids = torch.cat([current_ids, next_token_id], dim=-1)

        # Build final result
        output_text = "".join(all_tokens)
        visible_text = "".join(generated_tokens)

        return GenerationResult(
            output_text=output_text,
            visible_text=visible_text,
            state=state,
            prompt=prompt,
        )


@dataclass
class GenerationResult:
    """Container for SwiReasoning generation output.

    Attributes:
        output_text: Full generated text including ``<think>``/``</think>`` markers.
        visible_text: Only the tokens generated in explicit (visible) mode.
        state: Final :class:`GenerationState` with counters and entropy history.
        prompt: The original input prompt.
    """
    output_text: str
    visible_text: str
    state: GenerationState
    prompt: str

    def summary(self) -> dict[str, object]:
        """Return a summary dictionary of the generation run.

        Returns:
            Dictionary with key statistics: total tokens, thinking tokens,
            switch count, overthinking ratio, and entropy statistics.
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


class SwiReasoningSimulator:
    """Lightweight simulator for SwiReasoning without requiring a real model.

    Generates synthetic logit distributions with controllable entropy levels
    to demonstrate the mode-switching behavior. Useful for testing, demos,
    and environments where GPU models are unavailable.

    Example:
        >>> sim = SwiReasoningSimulator(vocab_size=1000, entropy_threshold=2.0)
        >>> result = sim.simulate("Test prompt", num_steps=20)
        >>> print(result.summary())
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        entropy_threshold: float = 2.0,
        max_switches: int = 3,
        max_thinking_tokens: int = 32,
        seed: Optional[int] = 42,
    ) -> None:
        """Initialize the simulator.

        Args:
            vocab_size: Size of the simulated vocabulary.
            entropy_threshold: Entropy threshold for mode switching.
            max_switches: Maximum mode switches allowed.
            max_thinking_tokens: Maximum consecutive thinking tokens.
            seed: Random seed for reproducibility. None for non-deterministic.
        """
        self.vocab_size = vocab_size
        self.entropy_threshold = entropy_threshold
        self.max_switches = max_switches
        self.max_thinking_tokens = max_thinking_tokens

        if seed is not None:
            torch.manual_seed(seed)
            self.rng = torch.Generator().manual_seed(seed)
        else:
            self.rng = torch.Generator()

    def _generate_synthetic_logits(self, step: int, num_steps: int) -> torch.Tensor:
        """Generate synthetic logits with varying entropy levels.

        Creates a pattern where entropy oscillates — starting low (confident),
        rising mid-sequence (uncertain), and dropping again at the end.

        Args:
            step: Current generation step index.
            num_steps: Total number of steps in the simulation.

        Returns:
            Tensor of shape ``(vocab_size,)`` with synthetic logit values.
        """
        import math

        # Create an oscillating confidence pattern
        phase = step / max(num_steps - 1, 1)

        # Entropy rises in the middle of the sequence and drops at the ends
        # This simulates: confident → uncertain → confident
        uncertainty = math.sin(phase * math.pi) * 2.5 + 0.5

        # Generate logits: low concentration = high entropy, high = low entropy
        concentration = max(0.1, 5.0 - uncertainty * 2.0)
        logits = torch.randn(self.vocab_size) * concentration

        # Make the top token more dominant when concentration is high
        if concentration > 2.0:
            top_idx = torch.randint(0, self.vocab_size, (1,)).item()
            logits[top_idx] += concentration * 3.0

        return logits

    def simulate(
        self,
        prompt: str,
        num_steps: int = 30,
    ) -> GenerationResult:
        """Run a simulated SwiReasoning generation.

        Produces synthetic logits at each step, computes entropy, and performs
        the same mode-switching logic as the real engine.

        Args:
            prompt: Input prompt text (used for display purposes).
            num_steps: Number of generation steps to simulate.

        Returns:
            A :class:`GenerationResult` with the simulation output.
        """
        state = GenerationState()
        all_tokens: list[str] = []
        visible_tokens: list[str] = []

        # Simulated token vocabulary for display
        sample_words = [
            "the", "model", "processes", "input", "data",
            "with", "neural", "network", "layers", "to",
            "generate", "accurate", "predictions", "using",
            "advanced", "reasoning", "capabilities", "and",
            "sophisticated", "algorithms", "for", "better",
            "understanding", "of", "complex", "patterns",
            "in", "natural", "language", "processing",
        ]

        logger.info("Starting SwiReasoning simulation: %d steps", num_steps)

        for step in range(num_steps):
            # Generate synthetic logits
            logits = self._generate_synthetic_logits(step, num_steps)
            entropy = SwiReasoningEngine.calculate_entropy(logits)
            state.entropy_history.append(entropy)

            # ─── State Machine (identical to real engine) ────────────
            if state.mode == InferenceMode.EXPLICIT:
                if (
                    entropy > self.entropy_threshold
                    and state.switch_count < self.max_switches
                ):
                    state.mode = InferenceMode.LATENT
                    state.switch_count += 1
                    state.thinking_token_count = 0
                    all_tokens.append(SwiReasoningEngine.THINK_START_TOKEN)
                    logger.debug(
                        "Step %d: → LATENT (H=%.3f > %.3f)", step, entropy,
                        self.entropy_threshold,
                    )

            elif state.mode == InferenceMode.LATENT:
                should_exit = (
                    entropy < self.entropy_threshold
                    or state.thinking_token_count >= self.max_thinking_tokens
                )
                if should_exit:
                    state.mode = InferenceMode.EXPLICIT
                    all_tokens.append(SwiReasoningEngine.THINK_END_TOKEN)
                    logger.debug(
                        "Step %d: → EXPLICIT (H=%.3f)", step, entropy,
                    )
                    state.thinking_token_count = 0

            # ─── Simulated Token Selection ───────────────────────────
            word_idx = step % len(sample_words)
            token = sample_words[word_idx]
            all_tokens.append(f" {token}")
            state.total_tokens += 1

            if state.mode == InferenceMode.LATENT:
                state.thinking_token_count += 1
                state.total_thinking_tokens += 1
            else:
                visible_tokens.append(f" {token}")

        # Close any open thinking block
        if state.mode == InferenceMode.LATENT:
            all_tokens.append(SwiReasoningEngine.THINK_END_TOKEN)

        return GenerationResult(
            output_text="".join(all_tokens),
            visible_text="".join(visible_tokens),
            state=state,
            prompt=prompt,
        )
