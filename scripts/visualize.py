"""
Generates visualizations from dataset statistics.
"""
import os
import matplotlib.pyplot as plt

def generate_visualizations(stats_report: dict, output_dir: str = "plots") -> list:
    os.makedirs(output_dir, exist_ok=True)
    raw = stats_report["_raw"]
    saved_paths = []
    def _save(fig, name):
        path = os.path.join(output_dir, name)
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        saved_paths.append(path)
    categories = stats_report["category_counts"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(categories.keys(), categories.values(), color="#4C72B0")
    ax.set_title("Examples per Category")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    _save(fig, "category_counts.png")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(raw["func_line_lens"], bins=range(1, max(raw["func_line_lens"]) + 2), color="#55A868")
    ax.set_title("Function Length Distribution (lines)")
    ax.set_xlabel("Lines of code")
    ax.set_ylabel("Frequency")
    _save(fig, "function_length_hist.png")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(raw["doc_line_lens"], bins=range(1, max(raw["doc_line_lens"]) + 2), color="#C44E52")
    ax.set_title("Docstring Length Distribution (lines)")
    ax.set_xlabel("Lines")
    ax.set_ylabel("Frequency")
    _save(fig, "docstring_length_hist.png")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(raw["func_token_counts"], bins=15, alpha=0.6, label="Function tokens", color="#4C72B0")
    ax.hist(raw["doc_token_counts"], bins=15, alpha=0.6, label="Docstring tokens", color="#DD8452")
    ax.set_title("Token Count Distribution")
    ax.set_xlabel("Token count")
    ax.set_ylabel("Frequency")
    ax.legend()
    _save(fig, "token_count_hist.png")
    param_dist = stats_report["parameter_distribution"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(param_dist.keys()), list(param_dist.values()), color="#8172B2")
    ax.set_title("Function Parameter Count Distribution")
    ax.set_xlabel("Number of parameters")
    ax.set_ylabel("Count")
    _save(fig, "parameter_count_dist.png")
    return saved_paths
