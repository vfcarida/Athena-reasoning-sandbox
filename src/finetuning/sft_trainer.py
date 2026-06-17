"""SFT Orchestrator — Supervised Fine-Tuning for instruction-following models.

This module provides a high-level wrapper around the Hugging Face ``trl.SFTTrainer``
for supervised fine-tuning of language models on instruction datasets. It supports:

- Multiple dataset formats (Alpaca, ChatML, ShareGPT)
- Custom formatting templates for instruction/input/output structure
- Multi-turn conversation handling
- Evaluation during training with validation splits
- Integration with PEFT adapters (LoRA/QLoRA) from :mod:`lora_trainer`

References:
    - Ouyang et al. (2022). Training language models to follow instructions.
    - Taori et al. (2023). Stanford Alpaca: An Instruction-following LLaMA Model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Format Templates
# ─────────────────────────────────────────────────────────────────────────────

ALPACA_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

ALPACA_NO_INPUT_TEMPLATE = """### Instruction:
{instruction}

### Response:
{output}"""

CHATML_TEMPLATE = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""


@dataclass
class SFTConfig:
    """Configuration for supervised fine-tuning.

    Attributes:
        model_name: HF model identifier or local path.
        dataset_format: Format of the instruction dataset.
        max_seq_length: Maximum sequence length for training.
        packing: Whether to pack multiple sequences into one.
        template: Custom formatting template string.
        epochs: Number of training epochs.
        batch_size: Per-device training batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        learning_rate: Peak learning rate.
        weight_decay: L2 regularization.
        warmup_ratio: Fraction of steps for warmup.
        max_grad_norm: Gradient clipping norm.
        mixed_precision: AMP dtype string.
        gradient_checkpointing: Use activation checkpointing.
        output_dir: Output directory for checkpoints.
        save_steps: Save checkpoint every N steps.
        log_steps: Log every N steps.
        seed: Random seed.
    """
    model_name: str = "gpt2"
    dataset_format: str = "alpaca"
    max_seq_length: int = 2048
    packing: bool = False
    template: Optional[str] = None
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 0.3
    mixed_precision: str = "no"
    gradient_checkpointing: bool = True
    output_dir: str = "./checkpoints/sft"
    save_steps: int = 200
    log_steps: int = 10
    seed: int = 42


class SFTOrchestrator:
    """Supervised Fine-Tuning orchestrator for instruction-following models.

    Provides a simplified interface for SFT that handles dataset formatting,
    model loading, and training execution. Can be used standalone or in
    combination with LoRA/QLoRA from :class:`LoRAFineTuner`.

    Example:
        >>> config = SFTConfig(model_name="gpt2", max_seq_length=512)
        >>> orchestrator = SFTOrchestrator(config)
        >>> train_data = [
        ...     {"instruction": "Translate to French", "input": "Hello", "output": "Bonjour"},
        ... ]
        >>> orchestrator.train(train_data)
    """

    def __init__(self, config: SFTConfig) -> None:
        """Initialize the SFT orchestrator.

        Args:
            config: SFT configuration.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(config.seed)

        logger.info(
            "SFTOrchestrator initialized: model=%s, format=%s, seq_length=%d",
            config.model_name, config.dataset_format, config.max_seq_length,
        )

    def format_sample(self, sample: dict[str, str]) -> str:
        """Format a single instruction sample using the configured template.

        Supports Alpaca, ChatML, and custom template formats.

        Args:
            sample: Dictionary with instruction data. Expected keys depend on
                the format: ``instruction``, ``input`` (optional), ``output``,
                ``system`` (optional for ChatML).

        Returns:
            Formatted text string ready for tokenization.
        """
        fmt = self.config.dataset_format.lower()

        if self.config.template:
            return self.config.template.format(**sample)

        if fmt == "alpaca":
            has_input = sample.get("input", "").strip()
            if has_input:
                return ALPACA_TEMPLATE.format(**sample)
            return ALPACA_NO_INPUT_TEMPLATE.format(**sample)

        elif fmt == "chatml":
            system = sample.get("system", "You are a helpful assistant.")
            return CHATML_TEMPLATE.format(system=system, **sample)

        elif fmt == "sharegpt":
            # ShareGPT format: multi-turn conversation
            conversations = sample.get("conversations", [])
            parts = []
            for turn in conversations:
                role = turn.get("from", turn.get("role", "user"))
                content = turn.get("value", turn.get("content", ""))
                parts.append(f"<|{role}|>\n{content}")
            return "\n".join(parts)

        else:
            # Fallback: concatenate all values
            return " ".join(str(v) for v in sample.values())

    def prepare_dataset(
        self,
        data: list[dict[str, str]],
        tokenizer: Any,
    ) -> Any:
        """Convert raw instruction data into a HF Dataset for SFT.

        Args:
            data: List of instruction dictionaries.
            tokenizer: The model's tokenizer.

        Returns:
            A ``datasets.Dataset`` ready for SFT training.
        """
        from datasets import Dataset as HFDataset

        # Format each sample
        formatted_texts = [self.format_sample(s) for s in data]

        # Create HF dataset
        dataset = HFDataset.from_dict({"text": formatted_texts})

        logger.info("SFT dataset prepared: %d samples", len(dataset))
        return dataset

    def load_model_and_tokenizer(
        self,
        peft_model: Optional[Any] = None,
    ) -> tuple[Any, Any]:
        """Load the base model and tokenizer for SFT.

        Args:
            peft_model: Optional pre-configured PEFT model (from LoRAFineTuner).
                If provided, this model is used instead of loading a fresh one.

        Returns:
            Tuple of (model, tokenizer).
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=False,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if peft_model is not None:
            logger.info("Using provided PEFT model for SFT.")
            return peft_model, tokenizer

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float32,
        )

        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        num_params = sum(p.numel() for p in model.parameters())
        logger.info("Model loaded for SFT: %.2fM parameters", num_params / 1e6)

        return model, tokenizer

    def train(
        self,
        train_data: list[dict[str, str]],
        eval_data: Optional[list[dict[str, str]]] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Execute supervised fine-tuning.

        Uses HF ``SFTTrainer`` from the ``trl`` library for training. Falls
        back to a manual training loop if ``trl`` is not available.

        Args:
            train_data: List of instruction dictionaries for training.
            eval_data: Optional list of instruction dictionaries for evaluation.
            model: Optional pre-loaded model. If None, loads from config.
            tokenizer: Optional pre-loaded tokenizer.

        Returns:
            Dictionary with training results and metrics.
        """
        if model is None or tokenizer is None:
            model, tokenizer = self.load_model_and_tokenizer()

        train_dataset = self.prepare_dataset(train_data, tokenizer)
        eval_dataset = self.prepare_dataset(eval_data, tokenizer) if eval_data else None

        try:
            return self._train_with_trl(model, tokenizer, train_dataset, eval_dataset)
        except ImportError:
            logger.warning("trl not available. Using manual SFT training loop.")
            return self._train_manual(model, tokenizer, train_dataset, eval_dataset)

    def _train_with_trl(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: Any,
        eval_dataset: Optional[Any],
    ) -> dict[str, Any]:
        """Train using TRL's SFTTrainer."""
        from trl import SFTTrainer, SFTConfig as TRLSFTConfig

        training_args = TRLSFTConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            max_grad_norm=self.config.max_grad_norm,
            logging_steps=self.config.log_steps,
            save_steps=self.config.save_steps,
            save_total_limit=3,
            max_seq_length=self.config.max_seq_length,
            packing=self.config.packing,
            seed=self.config.seed,
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )

        logger.info("Starting SFT training with TRL...")
        result = trainer.train()

        # Save final model
        trainer.save_model(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)

        logger.info("SFT training complete. Model saved to: %s", self.config.output_dir)

        return {
            "train_loss": result.training_loss,
            "train_runtime": result.metrics.get("train_runtime", 0),
            "output_dir": self.config.output_dir,
        }

    def _train_manual(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: Any,
        eval_dataset: Optional[Any],
    ) -> dict[str, Any]:
        """Manual SFT training loop (fallback when trl is unavailable)."""
        from torch.utils.data import DataLoader

        model = model.to(self.device)
        model.train()

        # Simple tokenized dataloader
        def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
            texts = [item["text"] for item in batch]
            encoded = tokenizer(
                texts,
                max_length=self.config.max_seq_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            encoded["labels"] = encoded["input_ids"].clone()
            return encoded

        loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        total_loss = 0.0
        steps = 0

        for epoch in range(self.config.epochs):
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                steps += 1

                if steps % self.config.log_steps == 0:
                    logger.info(
                        "SFT Step %d | Loss: %.4f", steps, total_loss / steps,
                    )

        # Save
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))

        return {
            "train_loss": total_loss / max(steps, 1),
            "total_steps": steps,
            "output_dir": self.config.output_dir,
        }

    @staticmethod
    def load_instruction_data(path: str | Path) -> list[dict[str, str]]:
        """Load instruction data from a JSONL file.

        Each line should be a JSON object with at least ``instruction`` and
        ``output`` keys (Alpaca format).

        Args:
            path: Path to the JSONL instruction file.

        Returns:
            List of instruction dictionaries.
        """
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        logger.info("Loaded %d instruction samples from: %s", len(data), path)
        return data
