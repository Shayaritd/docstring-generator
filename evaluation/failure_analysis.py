"""
Failure analysis: categorizes where fine-tuned model still struggles.
"""
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "core"))
from features import extract_features


def is_failure(per_example_result: Dict, bleu_threshold: float = 10.0) -> bool:
    sig = per_example_result["signature"]
    sections = per_example_result["sections"]
    if not sig["signature_match"]:
        return True
    if sections["sections_missing"]:
        return True
    if sections["exact_match_rate"] is not None and sections["exact_match_rate"] < 0.5:
        return True
    return False


def categorize_failures(per_example_results: List[Dict]) -> Dict:
    by_category = defaultdict(lambda: {"total": 0, "failures": 0})
    by_complexity = defaultdict(lambda: {"total": 0, "failures": 0})
    error_patterns = Counter()
    failure_examples = []

    for result in per_example_results:
        features = extract_features(result["code"])
        category = features.category
        bucket = features.complexity_bucket

        by_category[category]["total"] += 1
        by_complexity[bucket]["total"] += 1

        if is_failure(result):
            by_category[category]["failures"] += 1
            by_complexity[bucket]["failures"] += 1
            failure_examples.append(result)

            sig = result["signature"]
            sections = result["sections"]
            if sig["missing_params"]:
                error_patterns["missing_parameter(s)_in_Args"] += 1
            if sig["hallucinated_params"]:
                error_patterns["hallucinated_parameter(s)_in_Args"] += 1
            if "Raises" in sections["sections_missing"]:
                error_patterns["missing_Raises_section"] += 1
            if "Returns" in sections["sections_missing"]:
                error_patterns["missing_Returns_section"] += 1
            if "Args" in sections["sections_missing"]:
                error_patterns["missing_Args_section"] += 1
            if sections["sections_extra"]:
                error_patterns["extra_unexpected_sections"] += 1

    def _finalize(d):
        return {k: {**v, "failure_rate": v["failures"] / v["total"] if v["total"] else 0.0} for k, v in d.items()}

    return {
        "by_category": _finalize(by_category),
        "by_complexity": _finalize(by_complexity),
        "common_error_patterns": dict(error_patterns.most_common()),
        "failure_examples": failure_examples,
    }


def summarize_hardest_categories(failure_analysis: Dict, top_n: int = 3) -> List[str]:
    candidates = [(cat, stats["failure_rate"]) for cat, stats in failure_analysis["by_category"].items() if stats["total"] >= 2]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [f"{cat} ({rate:.0%} failure rate)" for cat, rate in candidates[:top_n]]
