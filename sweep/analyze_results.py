"""
Analyzes completed Optuna sweep results.

Usage:
    python analyze_results.py --storage sqlite:///sweep_results.db --study_name docstring_gen_sweep
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Optuna sweep results")
    parser.add_argument("--storage", type=str, default="sqlite:///sweep_results.db")
    parser.add_argument("--study_name", type=str, default="docstring_gen_sweep")
    parser.add_argument("--output_dir", type=str, default="sweep_analysis")
    parser.add_argument("--top_n", type=int, default=10)
    return parser.parse_args()


def load_study(storage: str, study_name: str):
    import optuna
    return optuna.load_study(study_name=study_name, storage=storage)


def build_comparison_table(study, top_n: int = 10) -> pd.DataFrame:
    df = study.trials_dataframe(attrs=("number", "value", "params", "state", "duration"))
    df = df[df["state"] == "COMPLETE"].copy()
    if df.empty:
        raise ValueError("No completed trials found.")
    df = df.sort_values("value", ascending=True)
    df = df.rename(columns={"value": "val_loss"})
    return df.head(top_n).reset_index(drop=True)


def get_best_config(study) -> dict:
    best = study.best_trial
    return {"params": best.params, "val_loss": best.value, "trial_number": best.number}


def plot_param_importance(study, output_dir: str) -> str:
    import optuna
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < 2:
        raise ValueError(f"Need at least 2 completed trials, found {len(completed)}")

    importances = optuna.importance.get_param_importances(study)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(list(importances.keys()), list(importances.values()), color="#4C72B0")
    ax.set_xlabel("Relative importance")
    ax.set_title("Hyperparameter Importance")
    ax.invert_yaxis()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "param_importance.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_optimization_history(study, output_dir: str) -> str:
    df = study.trials_dataframe(attrs=("number", "value", "state"))
    df = df[df["state"] == "COMPLETE"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df["number"], df["value"], alpha=0.6, color="#55A868", label="Trial value")
    running_best = df["value"].cummin()
    ax.plot(df["number"], running_best, color="#C44E52", label="Best so far")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Validation loss")
    ax.set_title("Optimization History")
    ax.legend()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "optimization_history.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    study = load_study(args.storage, args.study_name)

    print("Building comparison table...")
    table = build_comparison_table(study, top_n=args.top_n)
    os.makedirs(args.output_dir, exist_ok=True)
    table_path = os.path.join(args.output_dir, "comparison_table.csv")
    table.to_csv(table_path, index=False)
    print(table.to_string(index=False))
    print(f"Saved -> {table_path}")

    best = get_best_config(study)
    print(f"\nBest trial: #{best['trial_number']} | val_loss={best['val_loss']:.4f}")
    print(f"Best params: {best['params']}")

    print("\nPlotting parameter importance...")
    importance_path = plot_param_importance(study, args.output_dir)
    print(f"Saved -> {importance_path}")

    print("Plotting optimization history...")
    history_path = plot_optimization_history(study, args.output_dir)
    print(f"Saved -> {history_path}")


if __name__ == "__main__":
    main()
