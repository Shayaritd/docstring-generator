"""
Takes the best hyperparameters found by the sweep, writes a final config.yaml,
and trains the final model with them.

Usage:
    python train_best.py --storage sqlite:///sweep_results.db --study_name docstring_gen_sweep
"""

import argparse
import os
import subprocess
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
from search_space import merge_into_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train final model with best sweep hyperparameters")
    parser.add_argument("--storage", type=str, default="sqlite:///sweep_results.db")
    parser.add_argument("--study_name", type=str, default="docstring_gen_sweep")
    parser.add_argument("--base_config", type=str, default="../training/config.yaml")
    parser.add_argument("--output_config", type=str, default="../training/config_best.yaml")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def get_best_params(storage: str, study_name: str) -> dict:
    import optuna
    study = optuna.load_study(study_name=study_name, storage=storage)
    print(f"Best trial #{study.best_trial.number} | val_loss={study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params


def write_best_config(base_config_path: str, best_params: dict, output_path: str) -> dict:
    with open(base_config_path) as f:
        base_config = yaml.safe_load(f)

    final_config = merge_into_config(base_config, best_params)
    final_config["training"]["output_dir"] = "./checkpoints_best"
    final_config["wandb"]["run_name"] = "final_best_config"

    with open(output_path, "w") as f:
        yaml.safe_dump(final_config, f, sort_keys=False)

    return final_config


def main():
    args = parse_args()

    best_params = get_best_params(args.storage, args.study_name)
    final_config = write_best_config(args.base_config, best_params, args.output_config)
    print(f"\nWrote final config -> {args.output_config}")

    if args.dry_run:
        print("--dry_run set: skipping training launch.")
        return

    print("Launching final training run...")
    train_script = os.path.join(os.path.dirname(__file__), "..", "training", "train.py")
    result = subprocess.run(
        [sys.executable, train_script, "--config", args.output_config],
        cwd=os.path.dirname(train_script),
    )
    if result.returncode != 0:
        print("Final training run failed.", file=sys.stderr)
        sys.exit(result.returncode)

    print("\nFinal training complete. LoRA adapter saved under:")
    print(f"  {final_config['training']['output_dir']}/final_model")


if __name__ == "__main__":
    main()
