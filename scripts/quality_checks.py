"""
Quality checks for the docstring dataset.
"""
import ast
import hashlib
from typing import List, Dict
from features import extract_features

GOOGLE_SECTIONS = ("Args:", "Returns:", "Yields:", "Raises:", "Example:", "Examples:")

def check_google_style(docstring: str) -> List[str]:
    issues = []
    wrong_variants = {"Parameters:": "Args:", "Return:": "Returns:", "Raise:": "Raises:", "Yield:": "Yields:"}
    for wrong, correct in wrong_variants.items():
        if wrong in docstring:
            issues.append(f"Found '{wrong}' — should be '{correct}'")
    if not any(section in docstring for section in GOOGLE_SECTIONS):
        issues.append("No Google-style section headers found")
    return issues

def check_missing_sections(code: str, docstring: str) -> List[str]:
    issues = []
    features = extract_features(code, docstring)
    if features.has_raise and "Raises:" not in docstring:
        issues.append("Function raises but docstring has no 'Raises:' section")
    if features.has_yield and "Yields:" not in docstring:
        issues.append("Function yields but docstring has no 'Yields:' section")
    if features.has_yield and "Returns:" in docstring:
        issues.append("Generator should use 'Yields:' not 'Returns:'")
    documentable_params = features.num_params - (1 if features.is_method else 0)
    if documentable_params > 0 and "Args:" not in docstring:
        issues.append("Function has parameters but docstring has no 'Args:' section")
    return issues

def normalize_code(code: str) -> str:
    try:
        tree = ast.parse(code)
        return ast.dump(tree)
    except SyntaxError:
        return code.strip()

def find_duplicates(records: List[Dict[str, str]]) -> List[List[int]]:
    hash_to_indices: Dict[str, List[int]] = {}
    for i, r in enumerate(records):
        h = hashlib.sha256(normalize_code(r["input"]).encode("utf-8")).hexdigest()
        hash_to_indices.setdefault(h, []).append(i)
    return [indices for indices in hash_to_indices.values() if len(indices) > 1]

def run_quality_report(records: List[Dict[str, str]]) -> Dict:
    report = {"total_examples": len(records), "syntax_errors": [], "style_issues": {},
              "missing_section_issues": {}, "duplicate_groups": []}
    for i, r in enumerate(records):
        try:
            ast.parse(r["input"])
        except SyntaxError as e:
            report["syntax_errors"].append({"index": i, "error": str(e)})
            continue
        style_issues = check_google_style(r["output"])
        if style_issues:
            report["style_issues"][i] = style_issues
        missing = check_missing_sections(r["input"], r["output"])
        if missing:
            report["missing_section_issues"][i] = missing
    report["duplicate_groups"] = find_duplicates(records)
    report["clean_examples"] = (report["total_examples"] - len(report["syntax_errors"]) -
                               len(report["style_issues"]) - len(report["missing_section_issues"]))
    return report
