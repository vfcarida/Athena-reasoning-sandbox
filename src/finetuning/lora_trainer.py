"""LoRA/QLoRA Fine-Tuner — Parameter-Efficient Fine-Tuning with Low-Rank Adapters.

This module implements efficient fine-tuning of large language models using:

- **LoRA** (Low-Rank Adaptation): Injects trainable low-rank decomposition matrices
  into attention modules while freezing the base model weights. This dramatically
  reduces the number of trainable parameters (typically 0.1-1% of total).
- **QLoRA** (Quantized LoRA): Combines LoRA with 4-bit NF4 quantization of the
  base model weights via BitsAndBytes, enabling fine-tuning of 7B+ models on
  consumer-grade GPUs with as little as 12GB VRAM.

The adapter weights can be merged back into the base model to produce a
standalone model with zero inference overhead.

References:
    - Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models.
    - Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA/QLoRA fine-tuning.

    Attributes:
        model_name: HF model identifier or local path.
        rank: LoRA rank (r) — dimension of the low-rank decomposition.
            Higher rank = more capacity, more parameters. Typical: 8-64.
        alpha: LoRA alpha — scaling factor. Effective scaling = alpha/rank.
            Typical: 2*rank.
        dropout: Dropout applied to LoRA layers. Typical: 0.05-0.1.
        target_modules: List of module names to apply LoRA adapters to.
        bias: Bias training mode: "none", "all", "lora_only".
        task_type: PEFT task type: "CAUSAL_LM", "SEQ_2_SEQ_LM", etc.
        quantize_4bit: Whether to apply 4-bit quantization (QLoRA).
        quantization_type: Quantization type: "nf4" or "fp4".
        compute_dtype: Compute dtype for quantized layers.
        use_double_quant: Whether to use nested quantization.
        output_dir: Output directory for adapter weights.
        seed: Random seed.
    """
    model_name: str = "gpt2"
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    quantize_4bit: bool = False
    quantization_type: str = "nf4"
    compute_dtype: str = "bfloat16"
    use_double_quant: bool = True
    output_dir: str = "./checkpoints/lora"
    merged_output_dir: str = "./models/merged"
    seed: int = 42


class LoRAFineTuner:
    """Parameter-efficient fine-tuning using LoRA and QLoRA adapters.

    Applies low-rank decomposition matrices to specified attention modules,
    enabling efficient fine-tuning with a fraction of the original parameters.
    Supports optional 4-bit quantization (QLoRA) for memory-constrained
    environments.

    Example:
        >>> config = LoRAConfig(model_name="gpt2", rank=8, alpha=16)
        >>> tuner = LoRAFineTuner(config)
        >>> model, tokenizer = tuner.prepare_model()
        >>> # ... train with SFTOrchestrator ...
        >>> tuner.merge_and_save(model, tokenizer, "./merged_model")
    """

    def __init__(self, config: LoRAConfig) -> None:
        """Initialize the LoRA fine-tuner.

        Args:
            config: LoRA/QLoRA configuration.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(config.seed)

        logger.info(
            "LoRAFineTuner initialized: model=%s, rank=%d, alpha=%d, "
            "quantize=%s, targets=%s",
            config.model_name, config.rank, config.alpha,
            config.quantize_4bit, config.target_modules,
        )

    def prepare_model(self) -> tuple[nn.Module, Any]:
        """Load the base model and apply LoRA/QLoRA adapters.

        If ``quantize_4bit`` is enabled, the base model is loaded with
        BitsAndBytes 4-bit NF4 quantization before LoRA adapters are applied.
        Otherwise, the model is loaded in full precision.

        Returns:
            Tuple of (peft_model, tokenizer). The peft_model has LoRA adapters
            injected and only adapter parameters are trainable.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading model: %s", self.config.model_name)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=False,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model with optional quantization
        if self.config.quantize_4bit:
            model = self._load_quantized_model()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float32,
            )

        # Apply LoRA adapters
        peft_model = self.apply_lora(model)

        # Summary
        total_params = sum(p.numel() for p in peft_model.parameters())
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)

        logger.info(
            "LoRA applied: %.2fM total params, %.4fM trainable (%.2f%%)",
            total_params / 1e6,
            trainable_params / 1e6,
            100.0 * trainable_params / total_params,
        )

        return peft_model, tokenizer

    def _load_quantized_model(self) -> nn.Module:
        """Load the base model with 4-bit NF4 quantization (QLoRA).

        Uses BitsAndBytes for memory-efficient quantization that allows
        fine-tuning 7B+ parameter models on consumer GPUs.

        Returns:
            Quantized model ready for LoRA adapter injection.

        Raises:
            ImportError: If ``bitsandbytes`` is not installed.
        """
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        compute_dtype = dtype_map.get(self.config.compute_dtype, torch.bfloat16)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self.config.quantization_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.config.use_double_quant,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=False,
        )

        logger.info(
            "Model loaded with 4-bit quantization: type=%s, compute_dtype=%s",
            self.config.quantization_type, self.config.compute_dtype,
        )

        return model

    def apply_lora(self, model: nn.Module) -> nn.Module:
        """Inject LoRA adapters into the specified model modules.

        Creates low-rank decomposition matrices A and B for each target module:
            W' = W + (α/r) · B·A

        where W is the frozen original weight, A ∈ R^{d×r}, B ∈ R^{r×d},
        and only A and B are trainable.

        Args:
            model: The base model (possibly quantized) to adapt.

        Returns:
            PEFT-wrapped model with only LoRA parameters trainable.
        """
        from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType

        task_type_map = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
            "TOKEN_CLS": TaskType.TOKEN_CLS,
            "SEQ_CLS": TaskType.SEQ_CLS,
        }

        peft_config = PeftLoraConfig(
            r=self.config.rank,
            lora_alpha=self.config.alpha,
            lora_dropout=self.config.dropout,
            target_modules=self.config.target_modules,
            bias=self.config.bias,
            task_type=task_type_map.get(self.config.task_type, TaskType.CAUSAL_LM),
        )

        # Prepare model for k-bit training if quantized
        if self.config.quantize_4bit:
            try:
                from peft import prepare_model_for_kbit_training
                model = prepare_model_for_kbit_training(model)
                logger.info("Model prepared for k-bit training.")
            except ImportError:
                logger.warning("Could not prepare model for k-bit training.")

        peft_model = get_peft_model(model, peft_config)

        logger.info(
            "LoRA adapters applied: rank=%d, alpha=%d, targets=%s",
            self.config.rank, self.config.alpha, self.config.target_modules,
        )

        return peft_model

    def save_adapter(
        self,
        model: nn.Module,
        output_dir: Optional[str] = None,
    ) -> Path:
        """Save only the LoRA adapter weights to disk.

        The adapter weights are much smaller than the full model (typically
        a few MB vs. several GB), enabling efficient storage and sharing.

        Args:
            model: The PEFT model with trained LoRA adapters.
            output_dir: Output directory. Defaults to config output_dir.

        Returns:
            Path to the saved adapter directory.
        """
        path = Path(output_dir or self.config.output_dir)
        path.mkdir(parents=True, exist_ok=True)

        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(path))
        else:
            torch.save(model.state_dict(), path / "adapter_model.pt")

        logger.info("LoRA adapter saved to: %s", path)
        return path

    def merge_and_save(
        self,
        model: nn.Module,
        tokenizer: Any,
        output_dir: Optional[str] = None,
    ) -> Path:
        """Merge LoRA adapter weights back into the base model and save.

        Produces a standalone model with the adaptations baked into the
        original weight matrices. The merged model has zero inference
        overhead compared to the base model.

        Merge operation per adapted module:
            W_merged = W_base + (α/r) · B · A

        Args:
            model: The PEFT model with trained LoRA adapters.
            tokenizer: The model's tokenizer (saved alongside).
            output_dir: Output directory for the merged model.
                Defaults to config merged_output_dir.

        Returns:
            Path to the saved merged model directory.
        """
        path = Path(output_dir or self.config.merged_output_dir)
        path.mkdir(parents=True, exist_ok=True)

        if hasattr(model, "merge_and_unload"):
            logger.info("Merging LoRA adapters into base model...")
            merged_model = model.merge_and_unload()

            if hasattr(merged_model, "save_pretrained"):
                merged_model.save_pretrained(str(path))
            else:
                torch.save(merged_model.state_dict(), path / "pytorch_model.bin")
        else:
            logger.warning(
                "Model does not support merge_and_unload. "
                "Saving adapter weights only."
            )
            self.save_adapter(model, str(path))

        tokenizer.save_pretrained(str(path))

        logger.info("Merged model saved to: %s", path)
        return path

    @staticmethod
    def load_adapter(
        base_model_name: str,
        adapter_path: str | Path,
    ) -> tuple[nn.Module, Any]:
        """Load a previously saved LoRA adapter on top of a base model.

        Useful for inference or continued training from a saved adapter.

        Args:
            base_model_name: HF model identifier for the base model.
            adapter_path: Path to the saved adapter weights.

        Returns:
            Tuple of (peft_model, tokenizer).
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))

        logger.info(
            "Adapter loaded: base=%s, adapter=%s", base_model_name, adapter_path,
        )

        return peft_model, tokenizer

    def print_trainable_summary(self, model: nn.Module) -> dict[str, Any]:
        """Print and return a summary of trainable vs. total parameters.

        Args:
            model: The PEFT model to summarize.

        Returns:
            Dictionary with parameter counts and percentages.
        """
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable

        summary = {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
            "trainable_percentage": 100.0 * trainable / total if total > 0 else 0.0,
            "memory_saved_percentage": 100.0 * frozen / total if total > 0 else 0.0,
        }

        logger.info(
            "Parameter summary: Total=%.2fM, Trainable=%.4fM (%.2f%%), "
            "Frozen=%.2fM (%.2f%%)",
            total / 1e6, trainable / 1e6, summary["trainable_percentage"],
            frozen / 1e6, summary["memory_saved_percentage"],
        )

        return summary
