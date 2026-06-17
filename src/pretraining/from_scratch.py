"""From-Scratch Pretraining — Transformer initialization and training loop.

This module provides a complete pipeline for training a Transformer language model
from randomly initialized weights. It supports:

- Configurable architecture (GPT-2 style) with customizable dimensions
- BPE tokenizer training from domain-specific corpora
- Causal Language Modeling (CLM) training with AMP and gradient accumulation
- Cosine warmup learning rate scheduling
- Checkpoint saving/loading for training resumption

The primary use case is domain-specific pretraining where standard pretrained
representations fail (e.g., highly specialized financial, legal, or scientific
vocabularies as demonstrated by the BERTau model for financial NLP).

References:
    - Radford et al. (2019). Language Models are Unsupervised Multitask Learners.
    - BERTau: Domain-specific pretraining reducing fine-tuning needs by 66%.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass
class PretrainingConfig:
    """Configuration for from-scratch Transformer pretraining.

    Attributes:
        model_type: Base architecture type (e.g., "gpt2").
        vocab_size: Vocabulary size for the tokenizer.
        hidden_size: Hidden dimension (d_model) of the Transformer.
        num_hidden_layers: Number of Transformer layers.
        num_attention_heads: Number of multi-head attention heads.
        intermediate_size: FFN intermediate dimension.
        max_position_embeddings: Maximum context window length.
        epochs: Number of training epochs.
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        learning_rate: Peak learning rate.
        weight_decay: L2 regularization coefficient.
        warmup_ratio: Fraction of total steps for linear warmup.
        max_grad_norm: Maximum gradient norm for clipping.
        mixed_precision: AMP dtype string ("fp16", "bf16", "no").
        gradient_checkpointing: Whether to use activation checkpointing.
        output_dir: Directory for saving checkpoints.
        save_steps: Save a checkpoint every N optimization steps.
        save_total_limit: Maximum number of checkpoints to keep.
        log_steps: Log metrics every N steps.
        seed: Random seed for reproducibility.
    """
    model_type: str = "gpt2"
    vocab_size: int = 32000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 2048
    epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    mixed_precision: str = "no"
    gradient_checkpointing: bool = False
    output_dir: str = "./checkpoints/pretrain"
    save_steps: int = 500
    save_total_limit: int = 3
    log_steps: int = 10
    seed: int = 42


class TextDataset(Dataset):
    """Simple text dataset for causal language modeling.

    Tokenizes a list of text documents and chunks them into fixed-length
    sequences suitable for CLM training.

    Args:
        texts: List of text strings to use for training.
        tokenizer: Hugging Face tokenizer (PreTrainedTokenizerFast).
        max_length: Maximum sequence length per training sample.
    """

    def __init__(
        self,
        texts: list[str],
        tokenizer: Any,
        max_length: int = 2048,
    ) -> None:
        self.max_length = max_length
        self.examples: list[torch.Tensor] = []

        # Tokenize all texts and concatenate into one long sequence
        all_token_ids: list[int] = []
        for text in texts:
            encoded = tokenizer.encode(text)
            if isinstance(encoded, list):
                all_token_ids.extend(encoded)
            else:
                all_token_ids.extend(encoded.ids)

        # Chunk into fixed-length segments
        for i in range(0, len(all_token_ids) - max_length, max_length):
            chunk = all_token_ids[i : i + max_length + 1]
            if len(chunk) == max_length + 1:
                self.examples.append(torch.tensor(chunk, dtype=torch.long))

        logger.info(
            "TextDataset created: %d examples of length %d from %d total tokens",
            len(self.examples), max_length, len(all_token_ids),
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = self.examples[idx]
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }


class TransformerFromScratch:
    """Pipeline for training a Transformer language model from random initialization.

    Instantiates a GPT-2-style causal language model with fully configurable
    architecture hyperparameters, trains it on a text corpus using standard CLM
    loss, and supports checkpointing for fault-tolerant training.

    Example:
        >>> config = PretrainingConfig(hidden_size=256, num_hidden_layers=4)
        >>> trainer = TransformerFromScratch(config)
        >>> model = trainer.build_model()
        >>> tokenizer = trainer.build_tokenizer(["sample corpus text..."])
        >>> dataset = TextDataset(["training text..."], tokenizer, max_length=512)
        >>> trainer.train(dataset, model)
    """

    def __init__(self, config: PretrainingConfig) -> None:
        """Initialize the pretraining pipeline.

        Args:
            config: Pretraining configuration dataclass.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set seeds for reproducibility
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        logger.info(
            "TransformerFromScratch initialized: device=%s, hidden=%d, layers=%d, heads=%d",
            self.device, config.hidden_size, config.num_hidden_layers, config.num_attention_heads,
        )

    def build_model(self) -> nn.Module:
        """Instantiate a GPT-2-style causal language model from random weights.

        Creates a model using Hugging Face's ``GPT2LMHeadModel`` with a custom
        ``GPT2Config`` defined by the pipeline configuration. All parameters
        are randomly initialized (Xavier/He uniform depending on layer type).

        Returns:
            A ``GPT2LMHeadModel`` instance moved to the target device.
        """
        from transformers import GPT2Config, GPT2LMHeadModel

        model_config = GPT2Config(
            vocab_size=self.config.vocab_size,
            n_embd=self.config.hidden_size,
            n_layer=self.config.num_hidden_layers,
            n_head=self.config.num_attention_heads,
            n_inner=self.config.intermediate_size,
            n_positions=self.config.max_position_embeddings,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            bos_token_id=0,
            eos_token_id=1,
        )

        model = GPT2LMHeadModel(model_config)

        # Apply gradient checkpointing if configured
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(
            "Model built from scratch: %s, %.2fM parameters (%.2fM trainable)",
            model_config.model_type,
            num_params / 1e6,
            trainable_params / 1e6,
        )

        return model.to(self.device)

    def build_tokenizer(
        self,
        corpus_texts: list[str],
        save_path: Optional[str] = None,
    ) -> Any:
        """Train a BPE tokenizer from a domain-specific text corpus.

        Uses the Hugging Face ``tokenizers`` library to train a Byte-Pair
        Encoding tokenizer from scratch on the provided text corpus.

        Args:
            corpus_texts: List of text documents to train the tokenizer on.
            save_path: Optional path to save the trained tokenizer. If None,
                the tokenizer is only returned in memory.

        Returns:
            A trained ``PreTrainedTokenizerFast`` instance.
        """
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers
        from transformers import PreTrainedTokenizerFast

        # Initialize a BPE tokenizer
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # Configure the BPE trainer
        special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<think>", "</think>"]
        trainer = trainers.BpeTrainer(
            vocab_size=self.config.vocab_size,
            min_frequency=2,
            special_tokens=special_tokens,
            show_progress=True,
        )

        # Train on the provided corpus
        tokenizer.train_from_iterator(corpus_texts, trainer=trainer)

        # Wrap as a HF PreTrainedTokenizerFast
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
        )

        if save_path:
            path = Path(save_path)
            path.mkdir(parents=True, exist_ok=True)
            hf_tokenizer.save_pretrained(str(path))
            logger.info("Tokenizer saved to: %s", path)

        logger.info("Tokenizer trained: vocab_size=%d", len(hf_tokenizer))
        return hf_tokenizer

    def train(
        self,
        dataset: Dataset,
        model: nn.Module,
        validation_dataset: Optional[Dataset] = None,
    ) -> dict[str, list[float]]:
        """Execute the full pretraining loop with AMP and gradient accumulation.

        Implements causal language modeling (next-token prediction) training with:
        - Mixed precision training (AMP) when configured
        - Gradient accumulation for larger effective batch sizes
        - Cosine warmup learning rate scheduling
        - Gradient clipping for training stability
        - Periodic checkpoint saving
        - Validation loss tracking

        Args:
            dataset: Training dataset returning ``{"input_ids", "labels"}`` dicts.
            model: The model to train (from :meth:`build_model`).
            validation_dataset: Optional validation dataset for perplexity tracking.

        Returns:
            Dictionary with training history:
                - ``"train_loss"``: Per-step training losses.
                - ``"learning_rates"``: Per-step learning rates.
                - ``"val_loss"``: Per-evaluation validation losses (if validation provided).
        """
        cfg = self.config
        model.train()

        # DataLoader
        train_loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

        # Optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        # Learning rate scheduler with linear warmup + cosine decay
        total_steps = len(train_loader) * cfg.epochs // cfg.gradient_accumulation_steps
        warmup_steps = int(total_steps * cfg.warmup_ratio)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return current_step / max(warmup_steps, 1)
            progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)

        # Mixed precision setup
        use_amp = cfg.mixed_precision in ("fp16", "bf16")
        amp_dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16
        scaler = GradScaler(enabled=(cfg.mixed_precision == "fp16"))

        # Training history
        history: dict[str, list[float]] = {
            "train_loss": [],
            "learning_rates": [],
            "val_loss": [],
        }

        # Create output directory
        output_path = Path(cfg.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        global_step = 0
        optimizer.zero_grad()

        logger.info(
            "Starting pretraining: %d epochs, %d steps/epoch, %d total opt steps, "
            "warmup=%d steps, device=%s, AMP=%s",
            cfg.epochs, len(train_loader), total_steps, warmup_steps,
            self.device, cfg.mixed_precision,
        )

        for epoch in range(cfg.epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass with optional AMP
                with autocast(device_type=str(self.device), dtype=amp_dtype, enabled=use_amp):
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss / cfg.gradient_accumulation_steps

                # Backward pass
                if cfg.mixed_precision == "fp16":
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                epoch_loss += loss.item() * cfg.gradient_accumulation_steps
                num_batches += 1

                # Optimizer step (with accumulation)
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

                    # Record history
                    current_loss = loss.item() * cfg.gradient_accumulation_steps
                    current_lr = scheduler.get_last_lr()[0]
                    history["train_loss"].append(current_loss)
                    history["learning_rates"].append(current_lr)

                    # Logging
                    if global_step % cfg.log_steps == 0:
                        avg_loss = epoch_loss / num_batches
                        logger.info(
                            "Epoch %d | Step %d/%d | Loss: %.4f | LR: %.2e",
                            epoch + 1, global_step, total_steps, avg_loss, current_lr,
                        )

                    # Checkpoint saving
                    if global_step % cfg.save_steps == 0:
                        self.save_checkpoint(model, optimizer, scheduler, global_step, epoch)

            # End-of-epoch validation
            if validation_dataset is not None:
                val_loss = self._evaluate(model, validation_dataset, amp_dtype, use_amp)
                history["val_loss"].append(val_loss)
                perplexity = math.exp(min(val_loss, 20.0))
                logger.info(
                    "Epoch %d complete | Val Loss: %.4f | Perplexity: %.2f",
                    epoch + 1, val_loss, perplexity,
                )
            else:
                avg_loss = epoch_loss / max(num_batches, 1)
                logger.info(
                    "Epoch %d complete | Avg Train Loss: %.4f",
                    epoch + 1, avg_loss,
                )

        # Save final checkpoint
        self.save_checkpoint(model, optimizer, scheduler, global_step, cfg.epochs - 1, final=True)

        logger.info("Pretraining complete: %d total optimization steps.", global_step)
        return history

    def _evaluate(
        self,
        model: nn.Module,
        dataset: Dataset,
        amp_dtype: torch.dtype,
        use_amp: bool,
    ) -> float:
        """Compute average validation loss.

        Args:
            model: The model being trained.
            dataset: Validation dataset.
            amp_dtype: AMP data type.
            use_amp: Whether AMP is enabled.

        Returns:
            Average validation loss (float).
        """
        model.eval()
        val_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                with autocast(device_type=str(self.device), dtype=amp_dtype, enabled=use_amp):
                    outputs = model(input_ids=input_ids, labels=labels)

                total_loss += outputs.loss.item()
                num_batches += 1

        model.train()
        return total_loss / max(num_batches, 1)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        global_step: int,
        epoch: int,
        final: bool = False,
    ) -> Path:
        """Save a training checkpoint to disk.

        Serializes model weights, optimizer state, scheduler state, and training
        metadata for resumable training.

        Args:
            model: The model being trained.
            optimizer: The optimizer.
            scheduler: The learning rate scheduler.
            global_step: Current global optimization step.
            epoch: Current epoch number.
            final: Whether this is the final checkpoint.

        Returns:
            Path to the saved checkpoint directory.
        """
        tag = "final" if final else f"step-{global_step}"
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{tag}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "config": self.config,
        }, checkpoint_dir / "training_state.pt")

        # Also save the model in HF format for easy loading
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(checkpoint_dir))

        logger.info("Checkpoint saved: %s (step %d, epoch %d)", checkpoint_dir, global_step, epoch)

        # Enforce save_total_limit
        self._cleanup_old_checkpoints()

        return checkpoint_dir

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Load a training checkpoint for resumption.

        Args:
            checkpoint_path: Path to checkpoint directory or .pt file.
            model: Model to load weights into.
            optimizer: Optional optimizer to restore state.
            scheduler: Optional scheduler to restore state.

        Returns:
            Checkpoint metadata (global_step, epoch, config).
        """
        path = Path(checkpoint_path)
        state_file = path / "training_state.pt" if path.is_dir() else path

        checkpoint = torch.load(state_file, map_location=self.device, weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(
            "Checkpoint loaded: step=%d, epoch=%d",
            checkpoint.get("global_step", 0), checkpoint.get("epoch", 0),
        )

        return {
            "global_step": checkpoint.get("global_step", 0),
            "epoch": checkpoint.get("epoch", 0),
        }

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints exceeding ``save_total_limit``."""
        output_path = Path(self.config.output_dir)
        checkpoints = sorted(
            [d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda d: d.stat().st_mtime,
        )

        # Keep the final checkpoint and the N most recent
        while len(checkpoints) > self.config.save_total_limit:
            oldest = checkpoints.pop(0)
            if "final" not in oldest.name:
                import shutil
                shutil.rmtree(oldest)
                logger.info("Removed old checkpoint: %s", oldest)
