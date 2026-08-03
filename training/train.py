"""
Main training script for the docstring generator.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --resume_from_checkpoint auto
    python train.py --config config.yaml --resume_from_checkpoint ./checkpoints/checkpoint-300
    python train.py --config config.yaml --no_wandb
"""

import argparse
import os
import sys

import yaml

from data_utils import load_split_datasets, build_formatting_func, get_response_template
from callbacks import GPUMemoryLoggingCallback, SampleGenerationCallback, BLEUEvalCallback
from load_model import load_tokenizer, load_quantized_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train the docstring generator with QLoRA + SFTTrainer")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help="Path to a checkpoint dir to resume from, or 'auto' to resume from the "
             "latest checkpoint in output_dir if one exists"
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging even if enabled in config")
    parser.add_argument("--model", type=str, default=None, help="Override base_model from config")
    return parser.parse_args()


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_resume_checkpoint(resume_arg: str, output_dir: str):
    """Resolve the --resume_from_checkpoint argument to an actual path or None."""
    if resume_arg is None:
        return None
    if resume_arg == "auto":
        from transformers.trainer_utils import get_last_checkpoint
        if not os.path.isdir(output_dir):
            print(f"No output_dir found at {output_dir}; starting fresh instead of resuming.")
            return None
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is None:
            print(f"No checkpoint found in {output_dir}; starting fresh.")
            return None
        print(f"Resuming from latest checkpoint: {last_checkpoint}")
        return last_checkpoint
    if not os.path.isdir(resume_arg):
        raise FileNotFoundError(f"Checkpoint path does not exist: {resume_arg}")
    return resume_arg


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.model:
        config["model"]["base_model"] = args.model

    base_model = config["model"]["base_model"]
    wandb_enabled = config["wandb"]["enabled"] and not args.no_wandb

    # --- W&B setup ---
    if wandb_enabled:
        import wandb
        wandb.init(
            project=config["wandb"]["project"],
            entity=config["wandb"]["entity"],
            name=config["wandb"]["run_name"],
            config=config,
        )
        print(f"W&B run: {wandb.run.url}")
    else:
        print("W&B logging disabled.")

    # --- Model + tokenizer (reusing Phase 2 Day 3 loading code) ---
    print(f"\nLoading tokenizer and model: {base_model}")
    tokenizer = load_tokenizer(base_model)
    tokenizer.model_max_length = config["data"]["max_seq_length"]
    model = load_quantized_model(base_model)

    # --- Data ---
    print("\nLoading datasets...")
    dataset_dict = load_split_datasets(
        config["data"]["train_path"], config["data"]["val_path"], config["data"]["test_path"]
    )
    print(f"  Train: {len(dataset_dict['train'])} | Val: {len(dataset_dict['validation'])} | Test: {len(dataset_dict['test'])}")

    formatting_func = build_formatting_func(tokenizer)
    response_template = get_response_template(base_model)

    from trl import SFTTrainer, SFTConfig
    from transformers import DataCollatorForLanguageModeling, EarlyStoppingCallback
    try:
        from trl import DataCollatorForCompletionOnlyLM
    except ImportError:
        from transformers import DataCollatorForCompletionOnlyLM

    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # --- SFTConfig (superset of TrainingArguments in current TRL) ---
    t = config["training"]
    sft_config = SFTConfig(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        save_strategy=t["save_strategy"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        eval_strategy=t["eval_strategy"],
        logging_steps=t["logging_steps"],
        max_seq_length=config["data"]["max_seq_length"],
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to=["wandb"] if wandb_enabled else [],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # --- Callbacks ---
    val_examples = list(dataset_dict["validation"])
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=t["early_stopping_patience"]),
        GPUMemoryLoggingCallback(),
        SampleGenerationCallback(
            model, tokenizer, val_examples,
            num_samples=config["generation"]["num_sample_generations_per_epoch"],
            max_new_tokens=config["generation"]["max_new_tokens"],
            use_wandb=wandb_enabled,
        ),
        BLEUEvalCallback(
            model, tokenizer, val_examples,
            max_eval_examples=config["generation"]["max_bleu_eval_examples"],
            max_new_tokens=config["generation"]["max_new_tokens"],
            use_wandb=wandb_enabled,
        ),
    ]

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        formatting_func=formatting_func,
        data_collator=collator,
        callbacks=callbacks,
    )

    resume_checkpoint = resolve_resume_checkpoint(args.resume_from_checkpoint, t["output_dir"])

    # --- Train, with OOM and interrupt handling ---
    print("\nStarting training...")
    try:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current state before exiting...")
        trainer.save_model(os.path.join(t["output_dir"], "interrupted_checkpoint"))
        print(f"Saved to {t['output_dir']}/interrupted_checkpoint. Resume with --resume_from_checkpoint auto")
        sys.exit(0)
    except Exception as e:
        if "out of memory" in str(e).lower():
            print(
                f"\nOOM during training: {e}\n"
                "Try: reduce per_device_train_batch_size, increase "
                "gradient_accumulation_steps to compensate, or reduce max_seq_length."
            )
        raise

    # --- Final evaluation + save ---
    print("\nTraining complete. Running final evaluation on test set...")
    test_metrics = trainer.evaluate(eval_dataset=dataset_dict["test"])
    print("Test set metrics:", test_metrics)

    final_dir = os.path.join(t["output_dir"], "final_model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Final model saved to {final_dir}")

    if wandb_enabled:
        import wandb
        wandb.log({"test/" + k: v for k, v in test_metrics.items()})
        wandb.finish()


if __name__ == "__main__":
    main()
