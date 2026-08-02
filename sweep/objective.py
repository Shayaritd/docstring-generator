"""
Optuna objective function. Each trial:
  1. Samples hyperparameters
  2. Trains a short run with those hyperparameters
  3. Reports intermediate validation loss per epoch for pruning
  4. Logs the full trial to its own W&B run
  5. Returns final validation loss (the value Optuna minimizes)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

from search_space import suggest_hyperparameters, merge_into_config


def train_one_trial(config: dict, trial=None, use_wandb: bool = True, wandb_group: str = "sweep") -> float:
    from data_utils import load_split_datasets, build_formatting_func, get_response_template
    from load_model import load_tokenizer, load_quantized_model
    from training_config import LoRASettings, get_lora_config
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainerCallback
    try:
        from trl import DataCollatorForCompletionOnlyLM
    except ImportError:
        from transformers import DataCollatorForCompletionOnlyLM

    base_model = config["model"]["base_model"]
    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=config["wandb"]["project"],
            group=wandb_group,
            job_type="sweep_trial",
            name=f"trial_{trial.number}" if trial is not None else None,
            config=config,
            reinit=True,
        )

    try:
        tokenizer = load_tokenizer(base_model)
        model = load_quantized_model(base_model, attach_lora=False)

        lora_settings = LoRASettings(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["alpha"],
            lora_dropout=config["lora"]["dropout"],
            target_modules=config["lora"]["target_modules"],
        )
        lora_config = get_lora_config(lora_settings)

        dataset_dict = load_split_datasets(
            config["data"]["train_path"], config["data"]["val_path"], config["data"]["test_path"]
        )
        formatting_func = build_formatting_func(tokenizer)
        response_template = get_response_template(base_model)
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

        t = config["training"]
        sft_config = SFTConfig(
            output_dir=os.path.join(t["output_dir"], f"trial_{trial.number if trial else 'manual'}"),
            num_train_epochs=t["num_train_epochs"],
            per_device_train_batch_size=t["per_device_train_batch_size"],
            per_device_eval_batch_size=t["per_device_eval_batch_size"],
            learning_rate=t["learning_rate"],
            lr_scheduler_type=t["lr_scheduler_type"],
            warmup_ratio=t["warmup_ratio"],
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=t["logging_steps"],
            max_seq_length=config["data"]["max_seq_length"],
            bf16=True,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            report_to=["wandb"] if use_wandb else [],
        )

        class PruningCallback(TrainerCallback):
            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if trial is not None and metrics is not None and "eval_loss" in metrics:
                    trial.report(metrics["eval_loss"], step=int(state.epoch))
                    if trial.should_prune():
                        control.should_training_stop = True
                return control

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["validation"],
            formatting_func=formatting_func,
            data_collator=collator,
            peft_config=lora_config,
            callbacks=[PruningCallback()],
        )

        trainer.train()
        final_metrics = trainer.evaluate()
        val_loss = final_metrics["eval_loss"]

        if use_wandb:
            import wandb
            wandb.log({"final_eval_loss": val_loss})

        return val_loss

    except Exception as e:
        if "out of memory" in str(e).lower():
            raise RuntimeError(f"OOM in trial (batch_size={config['training']['per_device_train_batch_size']}): {e}") from e
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def build_objective(base_config: dict, use_wandb: bool = True):
    def objective(trial) -> float:
        sampled = suggest_hyperparameters(trial)
        trial_config = merge_into_config(base_config, sampled)

        try:
            val_loss = train_one_trial(trial_config, trial=trial, use_wandb=use_wandb)
        except RuntimeError as e:
            print(f"Trial {trial.number} failed: {e}")
            import optuna
            raise optuna.TrialPruned() from e

        return val_loss

    return objective
