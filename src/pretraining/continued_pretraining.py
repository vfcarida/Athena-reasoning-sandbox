"""Continued Pretraining — Domain adaptation of existing foundation models.

This module loads a pretrained open-weight model from Hugging Face Hub and
continues its pretraining on domain-specific corpora. Key capabilities:

- **Domain Adaptation**: Adjusts the model's probability distributions to
  accommodate domain-specific jargon, terminology, and document structures
  without erasing general language capabilities.
- **Context Window Extension**: Supports RoPE frequency scaling (linear,
  dynamic, YaRN) to extend the effective context length beyond the original
  pretraining window (e.g., 4K → 32K tokens).
- **Perplexity Tracking**: Monitors domain perplexity during training to
  ensure the model is learning the target distribution.

References:
    - Gururangan et al. (2020). Don't Stop Pretraining.
    - Chen et al. (2023). Extending Context Window of Large Language Models
      via Positional Interpolation.
    - Peng et al. (2023). YaRN: Efficient Context Window Extension of LLMs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass
class ContinuedPretrainingConfig:
    """Configuration for continued pretraining / domain adaptation.

    Attributes:
        base_model: HF model identifier or local path to the base model.
        rope_scaling_type: Type of RoPE scaling for context extension.
            Options: "linear", "dynamic", "yarn", None (no scaling).
        rope_scaling_factor: Factor by which to extend the context window.
            E.g., 4.0 extends 2048 → 8192.
        freeze_embeddings: Whether to freeze the embedding layer during training.
        epochs: Number of training epochs.
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        learning_rate: Peak learning rate (typically lower than from-scratch).
        weight_decay: L2 regularization.
        warmup_ratio: Fraction of total steps for warmup.
        max_grad_norm: Gradient clipping norm.
        mixed_precision: AMP mode ("fp16", "bf16", "no").
        gradient_checkpointing: Use activation checkpointing.
        max_seq_length: Maximum sequence length for training.
        output_dir: Checkpoint output directory.
        save_steps: Save checkpoint every N steps.
        log_steps: Log metrics every N steps.
        seed: Random seed.
    """
    base_model: str = "gpt2"
    rope_scaling_type: Optional[str] = None
    rope_scaling_factor: float = 1.0
    freeze_embeddings: bool = False
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    mixed_precision: str = "no"
    gradient_checkpointing: bool = False
    max_seq_length: int = 2048
    output_dir: str = "./checkpoints/continued"
    save_steps: int = 500
    log_steps: int = 10
    seed: int = 42


class DomainTextDataset(Dataset):
    """Dataset for continued pretraining on domain-specific text.

    Tokenizes text documents using the model's existing tokenizer and creates
    fixed-length CLM training sequences.

    Args:
        texts: List of domain text documents.
        tokenizer: The base model's tokenizer.
        max_length: Maximum sequence length.
    """

    def __init__(
        self,
        texts: list[str],
        tokenizer: Any,
        max_length: int = 2048,
    ) -> None:
        self.examples: list[torch.Tensor] = []
        self.max_length = max_length

        # Tokenize and concatenate
        all_ids: list[int] = []
        for text in texts:
            encoded = tokenizer(text, add_special_tokens=True, truncation=False)
            all_ids.extend(encoded["input_ids"])

        # Chunk into training sequences
        for i in range(0, len(all_ids) - max_length, max_length):
            chunk = all_ids[i : i + max_length + 1]
            if len(chunk) == max_length + 1:
                self.examples.append(torch.tensor(chunk, dtype=torch.long))

        logger.info(
            "DomainTextDataset: %d sequences of length %d from %d total tokens",
            len(self.examples), max_length, len(all_ids),
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = self.examples[idx]
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }


class ContinuedPretrainer:
    """Domain adaptation pipeline for existing foundation models.

    Loads a pretrained model from Hugging Face Hub, optionally extends its
    context window via RoPE scaling, and continues pretraining on domain
    text with causal language modeling loss.

    Example:
        >>> config = ContinuedPretrainingConfig(base_model="gpt2")
        >>> pretrainer = ContinuedPretrainer(config)
        >>> model, tokenizer = pretrainer.load_base_model()
        >>> dataset = DomainTextDataset(["domain text..."], tokenizer, max_length=512)
        >>> history = pretrainer.train(model, dataset)
    """

    def __init__(self, config: ContinuedPretrainingConfig) -> None:
        """Initialize the continued pretraining pipeline.

        Args:
            config: Continued pretraining configuration.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        logger.info(
            "ContinuedPretrainer initialized: base_model=%s, device=%s",
            config.base_model, self.device,
        )

    def load_base_model(self) -> tuple[nn.Module, Any]:
        """Load a pretrained model and tokenizer from Hugging Face Hub.

        Applies RoPE scaling configuration if specified, enabling context
        window extension beyond the original pretraining length.

        Returns:
            Tuple of (model, tokenizer).
        """
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading base model: %s", self.config.base_model)

        # Load model configuration
        model_config = AutoConfig.from_pretrained(self.config.base_model)

        # Apply RoPE scaling for context extension
        if self.config.rope_scaling_type and self.config.rope_scaling_factor > 1.0:
            self._apply_rope_scaling(model_config)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=False,
        )

        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            config=model_config,
            torch_dtype=torch.float32,
        )

        # Apply gradient checkpointing
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Optionally freeze embeddings
        if self.config.freeze_embeddings:
            self._freeze_embeddings(model)

        model = model.to(self.device)

        num_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(
            "Model loaded: %.2fM params (%.2fM trainable), context=%d",
            num_params / 1e6, trainable / 1e6,
            getattr(model_config, "max_position_embeddings", "unknown"),
        )

        return model, tokenizer

    def _apply_rope_scaling(self, config: Any) -> None:
        """Apply Rotary Position Embedding (RoPE) scaling to the model config.

        Modifies the model configuration to support extended context windows
        by adjusting the RoPE frequency base.

        Supported scaling types:
        - ``linear``: Linearly interpolates position embeddings.
        - ``dynamic``: NTK-aware dynamic scaling.
        - ``yarn``: YaRN efficient context extension.

        Args:
            config: Model configuration object to modify in-place.
        """
        scaling_type = self.config.rope_scaling_type
        factor = self.config.rope_scaling_factor

        if hasattr(config, "rope_scaling"):
            config.rope_scaling = {
                "type": scaling_type,
                "factor": factor,
            }

            # Extend max position embeddings
            original_max = getattr(config, "max_position_embeddings", 2048)
            config.max_position_embeddings = int(original_max * factor)

            logger.info(
                "RoPE scaling applied: type=%s, factor=%.1f, context=%d → %d",
                scaling_type, factor, original_max, config.max_position_embeddings,
            )
        else:
            logger.warning(
                "Model config does not support rope_scaling. "
                "Context window extension may not work."
            )

    @staticmethod
    def _freeze_embeddings(model: nn.Module) -> None:
        """Freeze the embedding layer parameters.

        Args:
            model: The model whose embeddings to freeze.
        """
        frozen_count = 0
        for name, param in model.named_parameters():
            if "embed" in name.lower() or "wte" in name.lower() or "wpe" in name.lower():
                param.requires_grad = False
                frozen_count += 1

        logger.info("Froze %d embedding parameter groups.", frozen_count)

    def extend_context_window(
        self,
        model: nn.Module,
        target_length: int,
        rope_scaling_type: str = "linear",
    ) -> nn.Module:
        """Extend the model's context window via RoPE frequency adjustment.

        This is a convenience method that applies RoPE scaling after model
        loading. For models loaded with scaling already configured, this
        step is unnecessary.

        Args:
            model: The loaded model to modify.
            target_length: Desired maximum context length.
            rope_scaling_type: Scaling type to apply.

        Returns:
            The modified model (in-place modification).
        """
        if hasattr(model.config, "max_position_embeddings"):
            original = model.config.max_position_embeddings
            factor = target_length / original

            model.config.rope_scaling = {
                "type": rope_scaling_type,
                "factor": factor,
            }
            model.config.max_position_embeddings = target_length

            logger.info(
                "Context window extended: %d → %d (factor=%.2f, type=%s)",
                original, target_length, factor, rope_scaling_type,
            )

        return model

    def prepare_domain_dataset(
        self,
        texts: list[str],
        tokenizer: Any,
    ) -> DomainTextDataset:
        """Prepare a domain-specific dataset for continued pretraining.

        Args:
            texts: List of domain text documents.
            tokenizer: The model's tokenizer.

        Returns:
            A :class:`DomainTextDataset` ready for training.
        """
        return DomainTextDataset(
            texts=texts,
            tokenizer=tokenizer,
            max_length=self.config.max_seq_length,
        )

    def train(
        self,
        model: nn.Module,
        dataset: Dataset,
        validation_dataset: Optional[Dataset] = None,
    ) -> dict[str, list[float]]:
        """Execute continued pretraining with domain adaptation.

        The training loop is similar to from-scratch pretraining but uses a
        lower learning rate to avoid catastrophic forgetting of general
        language capabilities.

        Args:
            model: The pretrained model to continue training.
            dataset: Domain-specific training dataset.
            validation_dataset: Optional validation dataset.

        Returns:
            Dictionary with training history (train_loss, val_loss, learning_rates).
        """
        cfg = self.config
        model.train()

        train_loader = DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        )

        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        total_steps = len(train_loader) * cfg.epochs // cfg.gradient_accumulation_steps
        warmup_steps = int(total_steps * cfg.warmup_ratio)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)

        use_amp = cfg.mixed_precision in ("fp16", "bf16")
        amp_dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16
        scaler = GradScaler(enabled=(cfg.mixed_precision == "fp16"))

        history: dict[str, list[float]] = {
            "train_loss": [],
            "learning_rates": [],
            "val_loss": [],
        }

        output_path = Path(cfg.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        global_step = 0
        optimizer.zero_grad()

        logger.info(
            "Starting continued pretraining: %d epochs, %d total steps, "
            "lr=%.2e, device=%s",
            cfg.epochs, total_steps, cfg.learning_rate, self.device,
        )

        for epoch in range(cfg.epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # [EDUCATIONAL] Forward pass: Compute the next-token prediction loss
                # We use torch.autocast to automatically map operations to the appropriate
                # lower-precision dtype (e.g., bf16) to accelerate training.
                with autocast(device_type=self.device.type, dtype=amp_dtype, enabled=use_amp):
                    outputs = model(input_ids=input_ids, labels=labels)
                    # [EDUCATIONAL] Scale loss by accumulation steps so that the accumulated gradient
                    # matches the scale of the intended batch size.
                    loss = outputs.loss / cfg.gradient_accumulation_steps

                # [EDUCATIONAL] Backward pass: Backpropagate the gradients
                # Loss scaling is critical for fp16 to avoid underflow of tiny gradients.
                if cfg.mixed_precision == "fp16":
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                epoch_loss += loss.item() * cfg.gradient_accumulation_steps
                num_batches += 1

                if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                    if cfg.mixed_precision == "fp16":
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    current_loss = loss.item() * cfg.gradient_accumulation_steps
                    history["train_loss"].append(current_loss)
                    history["learning_rates"].append(scheduler.get_last_lr()[0])

                    if global_step % cfg.log_steps == 0:
                        perplexity = math.exp(min(epoch_loss / num_batches, 20.0))
                        logger.info(
                            "Epoch %d | Step %d/%d | Loss: %.4f | PPL: %.2f | LR: %.2e",
                            epoch + 1, global_step, total_steps,
                            epoch_loss / num_batches, perplexity,
                            scheduler.get_last_lr()[0],
                        )

                    if global_step % cfg.save_steps == 0:
                        ckpt_dir = output_path / f"checkpoint-step-{global_step}"
                        ckpt_dir.mkdir(parents=True, exist_ok=True)
                        if hasattr(model, "save_pretrained"):
                            model.save_pretrained(str(ckpt_dir))
                        logger.info("Checkpoint saved: %s", ckpt_dir)

            # End-of-epoch validation
            if validation_dataset is not None:
                val_loss = self._evaluate(model, validation_dataset, amp_dtype, use_amp)
                history["val_loss"].append(val_loss)
                logger.info(
                    "Epoch %d | Val Loss: %.4f | Val PPL: %.2f",
                    epoch + 1, val_loss, math.exp(min(val_loss, 20.0)),
                )

        # Save final model
        final_dir = output_path / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(final_dir))
        logger.info("Continued pretraining complete. Final model saved to: %s", final_dir)

        return history

    def _evaluate(
        self,
        model: nn.Module,
        dataset: Dataset,
        amp_dtype: torch.dtype,
        use_amp: bool,
    ) -> float:
        """Compute validation loss and perplexity."""
        model.eval()
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                with autocast(device_type=self.device.type, dtype=amp_dtype, enabled=use_amp):
                    outputs = model(input_ids=input_ids, labels=labels)
                total_loss += outputs.loss.item()
                count += 1

        model.train()
        return total_loss / max(count, 1)
