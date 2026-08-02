"""
Computes dataset statistics.
"""
import re
from collections import Counter
from features import extract_features

def compute_statistics(records: list, tokenizer_model: str = "regex_estimate") -> dict:
    categories = Counter()
    complexities = Counter()
    func_line_lens = []
    doc_line_lens = []
    func_token_counts = []
    doc_token_counts = []
    param_counts = Counter()
    raw = {"func_line_lens": [], "doc_line_lens": [], "func_token_counts": [], "doc_token_counts": []}
    for r in records:
        features = extract_features(r["input"], r["output"])
        categories[features.category] += 1
        complexities[features.complexity_bucket] += 1
        func_lines = len(r["input"].strip().splitlines())
        doc_lines = len(r["output"].strip().splitlines())
        func_line_lens.append(func_lines)
        doc_line_lens.append(doc_lines)
        raw["func_line_lens"].append(func_lines)
        raw["doc_line_lens"].append(doc_lines)
        func_tokens = len(r["input"].split())
        doc_tokens = len(r["output"].split())
        func_token_counts.append(func_tokens)
        doc_token_counts.append(doc_tokens)
        raw["func_token_counts"].append(func_tokens)
        raw["doc_token_counts"].append(doc_tokens)
        param_counts[features.num_params] += 1
    return {"category_counts": dict(categories), "complexity_counts": dict(complexities),
            "function_length": {"min": min(func_line_lens), "max": max(func_line_lens),
                               "mean": sum(func_line_lens) / len(func_line_lens),
                               "median": sorted(func_line_lens)[len(func_line_lens)//2]},
            "docstring_length": {"min": min(doc_line_lens), "max": max(doc_line_lens),
                                "mean": sum(doc_line_lens) / len(doc_line_lens),
                                "median": sorted(doc_line_lens)[len(doc_line_lens)//2]},
            "parameter_distribution": dict(param_counts),
            "tokenizer_used": tokenizer_model,
            "_raw": raw}
