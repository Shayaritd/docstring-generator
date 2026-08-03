"""
Central configuration for QLoRA fine-tuning of the docstring generator.

Import from here rather than hardcoding hyperparameters in training scripts,
so Phase 3 (hyperparameter tuning) has a single place to override values.
"""

from dataclasses import dataclass, field
from typing import List

import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig, TrainingArguments


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

# Pick one. Qwen2.5-Coder-1.5B is code-pretrained (usually better starting
# point for this task); Phi-3-mini-4k-instruct is a strong general instruct
# model. Both fit comfortably in 8-16GB VRAM under 4-bit quantization.
MODEL_OPTIONS = {
    "qwen": "Qwen/Qwen2.5-Coder-1.5B",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
}
BASE_MODEL = MODEL_OPTIONS["qwen"]


# ---------------------------------------------------------------------------
# Quantization (QLoRA, 4-bit)
# ---------------------------------------------------------------------------

def get_bnb_config() -> BitsAndBytesConfig:
    """4-bit quantization config with bf16 compute dtype (QLoRA paper defaults)."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


# ---------------------------------------------------------------------------
# LoRA adapter configuration
# ---------------------------------------------------------------------------

@dataclass
class LoRASettings:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


def get_lora_config(settings: LoRASettings = LoRASettings()) -> LoraConfig:
    """Build a peft LoraConfig from LoRASettings."""
    return LoraConfig(
        r=settings.r,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        target_modules=settings.target_modules,
        bias=settings.bias,
        task_type=settings.task_type,
    )


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class TrainSettings:
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    num_train_epochs: int = 3
    gradient_accumulation_steps: int = 1
    output_dir: str = "./checkpoints"
    logging_steps: int = 10
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 512


def get_training_arguments(settings: TrainSettings = TrainSettings()) -> TrainingArguments:
    """Build TrainingArguments with memory optimization defaults applied."""
    return TrainingArguments(
        output_dir=settings.output_dir,
        learning_rate=settings.learning_rate,
        per_device_train_batch_size=settings.per_device_train_batch_size,
        per_device_eval_batch_size=settings.per_device_eval_batch_size,
        num_train_epochs=settings.num_train_epochs,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        logging_steps=settings.logging_steps,
        eval_strategy=settings.eval_strategy,
        save_strategy=settings.save_strategy,
        save_total_limit=settings.save_total_limit,
        warmup_ratio=settings.warmup_ratio,
        lr_scheduler_type=settings.lr_scheduler_type,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
