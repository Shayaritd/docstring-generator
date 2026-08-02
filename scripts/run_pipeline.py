"""
Runs the full Day 2 pipeline end-to-end.
"""
import json
import sys
from data_loader import load_jsonl, save_jsonl
from augmentation import augment_example
from quality_checks import run_quality_report
from split_dataset import stratified_split, split_summary
from stats import compute_statistics
from visualize import generate_visualizations

def main(input_path: str):
    print(f"Loading dataset from {input_path}...")
    records = load_jsonl(input_path)
    print(f"Loaded {len(records)} examples.\n")
    print("Running quality checks...")
    quality_report = run_quality_report(records)
    print(f"  Clean examples:        {quality_report['clean_examples']}/{quality_report['total_examples']}")
    print(f"  Syntax errors:         {len(quality_report['syntax_errors'])}")
    print(f"  Style issues:          {len(quality_report['style_issues'])}")
    print(f"  Missing-section issues:{len(quality_report['missing_section_issues'])}")
    print(f"  Duplicate groups:      {len(quality_report['duplicate_groups'])}")
    with open("quality_report.json", "w") as f:
        json.dump(quality_report, f, indent=2)
    print("  Saved -> quality_report.json\n")
    print("Augmenting dataset...")
    augmented = []
    for r in records:
        augmented.extend(augment_example(r))
    print(f"  Original examples:  {len(records)}")
    print(f"  New augmented:      {len(augmented)}")
    full_dataset = records + augmented
    print(f"  Total after augment:{len(full_dataset)}\n")
    save_jsonl(full_dataset, "dataset_augmented.jsonl")
    print("  Saved -> dataset_augmented.jsonl\n")
    print("Creating stratified 80/10/10 split...")
    train, val, test = stratified_split(full_dataset)
    save_jsonl(train, "train.jsonl")
    save_jsonl(val, "val.jsonl")
    save_jsonl(test, "test.jsonl")
    summary = split_summary(train, val, test)
    print(json.dumps(summary, indent=2))
    print("  Saved -> train.jsonl, val.jsonl, test.jsonl\n")
    print("Computing statistics...")
    stats_report = compute_statistics(full_dataset)
    with open("dataset_stats.json", "w") as f:
        json.dump({k: v for k, v in stats_report.items() if k != "_raw"}, f, indent=2)
    print("  Saved -> dataset_stats.json\n")
    print("Generating visualizations...")
    paths = generate_visualizations(stats_report, output_dir="plots")
    for p in paths:
        print(f"  Saved -> {p}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "dataset.jsonl"
    main(path)
