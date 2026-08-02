"""
Generates human-friendly Markdown report from raw_results.json.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from failure_analysis import categorize_failures, summarize_hardest_categories, is_failure


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--output", type=str, default="eval_results/report.md")
    parser.add_argument("--num_highlight_examples", type=int, default=3)
    return parser.parse_args()


def _fmt(value, decimals=2):
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}" if isinstance(value, float) else str(value)


def build_summary_table(base: dict, finetuned: dict) -> str:
    rows = [
        ("BLEU", base["bleu"]["bleu"], finetuned["bleu"]["bleu"]),
        ("ROUGE-L F1", base["rouge_l"]["rouge_l_f1"], finetuned["rouge_l"]["rouge_l_f1"]),
        ("Section exact-match rate", base["avg_section_exact_match_rate"], finetuned["avg_section_exact_match_rate"]),
        ("Signature match rate", base["signature_match_rate"], finetuned["signature_match_rate"]),
    ]
    if "judge_scores" in base and "judge_scores" in finetuned:
        for dim in ("accuracy", "completeness", "clarity", "style"):
            base_scores = [s[dim] for s in base["judge_scores"] if s.get(dim) is not None]
            ft_scores = [s[dim] for s in finetuned["judge_scores"] if s.get(dim) is not None]
            base_avg = sum(base_scores) / len(base_scores) if base_scores else None
            ft_avg = sum(ft_scores) / len(ft_scores) if ft_scores else None
            rows.append((f"Judge: {dim} (1-5)", base_avg, ft_avg))
    lines = ["| Metric | Base (zero-shot) | Fine-tuned | Delta |", "|---|---|---|---|"]
    for name, base_val, ft_val in rows:
        delta = ""
        if isinstance(base_val, (int, float)) and isinstance(ft_val, (int, float)):
            delta = f"{ft_val - base_val:+.2f}"
        lines.append(f"| {name} | {_fmt(base_val, 3)} | {_fmt(ft_val, 3)} | {delta} |")
    return "\n".join(lines)


def build_side_by_side(examples_base: list, examples_ft: list, n: int) -> str:
    lines = ["## Side-by-Side Comparison (sample)\n"]
    for i in range(min(n, len(examples_base))):
        b, f = examples_base[i], examples_ft[i]
        lines.append(f"### Example {i + 1}\n")
        lines.append(f"**Code:**\n```python\n{b['code']}\n```\n")
        lines.append(f"**Reference:**\n```\n{b['reference']}\n```\n")
        lines.append(f"**Base (zero-shot):**\n```\n{b['generated']}\n```\n")
        lines.append(f"**Fine-tuned:**\n```\n{f['generated']}\n```\n")
        lines.append("---\n")
    return "\n".join(lines)


def find_shines_and_both_fail(base_per_example: list, ft_per_example: list) -> tuple:
    shines, both_fail = [], []
    for b, f in zip(base_per_example, ft_per_example):
        base_failed = is_failure(b)
        ft_failed = is_failure(f)
        if base_failed and not ft_failed:
            shines.append((b, f))
        elif base_failed and ft_failed:
            both_fail.append((b, f))
    return shines, both_fail


def build_highlight_section(title: str, pairs: list, n: int) -> str:
    if not pairs:
        return f"## {title}\n\n_None found in this test set._\n"
    lines = [f"## {title}\n"]
    for b, f in pairs[:n]:
        lines.append(f"**Code:**\n```python\n{b['code']}\n```\n")
        lines.append(f"**Reference:** {b['reference'][:150]}...\n")
        lines.append(f"**Base:** {b['generated'][:150]}...\n")
        lines.append(f"**Fine-tuned:** {f['generated'][:150]}...\n")
        lines.append("---\n")
    return "\n".join(lines)


def build_failure_analysis_section(finetuned_per_example: list) -> str:
    analysis = categorize_failures(finetuned_per_example)
    lines = ["## Failure Analysis (fine-tuned model)\n"]
    lines.append("### Failure rate by function category\n")
    lines.append("| Category | Total | Failures | Failure rate |")
    lines.append("|---|---|---|---|")
    for cat, stats in sorted(analysis["by_category"].items(), key=lambda x: -x[1]["failure_rate"]):
        lines.append(f"| {cat} | {stats['total']} | {stats['failures']} | {stats['failure_rate']:.0%} |")
    lines.append("\n### Failure rate by complexity\n")
    lines.append("| Complexity | Total | Failures | Failure rate |")
    lines.append("|---|---|---|---|")
    for bucket, stats in analysis["by_complexity"].items():
        lines.append(f"| {bucket} | {stats['total']} | {stats['failures']} | {stats['failure_rate']:.0%} |")
    lines.append("\n### Common error patterns\n")
    if analysis["common_error_patterns"]:
        for pattern, count in analysis["common_error_patterns"].items():
            lines.append(f"- **{pattern.replace('_', ' ')}**: {count} occurrence(s)")
    else:
        lines.append("- No recurring error patterns detected.")
    hardest = summarize_hardest_categories(analysis)
    lines.append(f"\n### Hardest function types\n")
    if hardest:
        for h in hardest:
            lines.append(f"- {h}")
    else:
        lines.append("- Not enough examples per category to identify a clear pattern.")
    return "\n".join(lines) + "\n"


def generate_report(results: dict, output_path: str, n_highlight: int = 3):
    base, finetuned = results["base"], results["finetuned"]
    sections = [
        "# Docstring Generator — Evaluation Report\n",
        f"Test set size: {len(base['per_example'])} held-out examples\n",
        "## Summary Statistics\n",
        build_summary_table(base, finetuned),
        "\n",
    ]
    shines, both_fail = find_shines_and_both_fail(base["per_example"], finetuned["per_example"])
    sections.append(f"Fine-tuning improved {len(shines)}/{len(base['per_example'])} previously-failing examples. "
                     f"{len(both_fail)} example(s) still fail in both models.\n")
    sections.append(build_highlight_section("Where Fine-Tuning Shines", shines, n_highlight))
    sections.append(build_highlight_section("Where Both Models Still Fail", both_fail, n_highlight))
    sections.append(build_failure_analysis_section(finetuned["per_example"]))
    sections.append(build_side_by_side(base["per_example"], finetuned["per_example"], n_highlight))
    report_text = "\n".join(sections)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_text)
    return report_text


def main():
    args = parse_args()
    with open(args.results) as f:
        results = json.load(f)
    report_text = generate_report(results, args.output, args.num_highlight_examples)
    print(f"Report saved -> {args.output}")


if __name__ == "__main__":
    main()
