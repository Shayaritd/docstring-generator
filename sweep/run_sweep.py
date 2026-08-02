"""
Runs the Optuna hyperparameter sweep.

Usage:
    python run_sweep.py --n_trials 20
    CUDA_VISIBLE_DEVICES=0 python run_sweep.py --n_trials 20 &
    CUDA_VISIBLE_DEVICES=1 python run_sweep.py --n_trials 20 &
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
from objective import build_objective


def parse_args():
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter sweep")
    parser.add_argument("--config", type=str, default="../training/config.yaml", help="Base config.yaml")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of trials to run in this process")
    parser.add_argument("--study_name", type=str, default="docstring_gen_sweep")
    parser.add_argument("--storage", type=str, default="sqlite:///sweep_results.db")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--no_pruning", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    import optuna

    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1) if not args.no_pruning else optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="minimize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    print(f"Study '{args.study_name}' — {len(study.trials)} trials already completed.")
    print(f"Running {args.n_trials} more trials in this process...")

    objective = build_objective(base_config, use_wandb=not args.no_wandb)

    try:
        study.optimize(objective, n_trials=args.n_trials, catch=(RuntimeError,))
    except KeyboardInterrupt:
        print("\nSweep interrupted. Progress is saved — rerun to resume.")
        sys.exit(0)

    print("\n=== Sweep complete ===")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best value (val_loss): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()
