"""
Structured docstring evaluation: section exact-match and signature detection.
"""
import ast
import re
from typing import Dict, List, Optional

SECTION_NAMES = ["Args", "Returns", "Raises", "Yields"]


def extract_sections(docstring: str) -> Dict[str, str]:
    pattern = r"^(" + "|".join(SECTION_NAMES) + r"):\s*$"
    lines = docstring.strip().splitlines()
    sections: Dict[str, List[str]] = {"Summary": []}
    current = "Summary"
    for line in lines:
        match = re.match(pattern, line.strip())
        if match:
            current = match.group(1)
            sections[current] = []
        else:
            sections.setdefault(current, []).append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}


def compare_sections(reference_doc: str, generated_doc: str) -> Dict[str, object]:
    ref_sections = extract_sections(reference_doc)
    gen_sections = extract_sections(generated_doc)

    def normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    result = {}
    matches = 0
    comparable = [s for s in ref_sections if s != "Summary"]
    for section in comparable:
        ref_text = normalize(ref_sections.get(section, ""))
        gen_text = normalize(gen_sections.get(section, ""))
        is_match = section in gen_sections and ref_text == gen_text
        result[f"{section}_exact_match"] = is_match
        if is_match:
            matches += 1
    result["sections_missing"] = [s for s in comparable if s not in gen_sections]
    result["sections_extra"] = [s for s in gen_sections if s != "Summary" and s not in ref_sections]
    result["exact_match_rate"] = matches / len(comparable) if comparable else None
    return result


def get_actual_parameters(code: str) -> Optional[List[str]]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = [a.arg for a in node.args.args if a.arg not in ("self", "cls")]
            if node.args.vararg:
                params.append(node.args.vararg.arg)
            if node.args.kwarg:
                params.append(node.args.kwarg.arg)
            return params
    return None


def check_signature_detection(code: str, generated_doc: str) -> Dict[str, object]:
    actual_params = get_actual_parameters(code) or []
    sections = extract_sections(generated_doc)
    args_text = sections.get("Args", "")
    documented_params = re.findall(r"^\s*(\w+)\s*(?:\([\w\[\], .]+\))?\s*:", args_text, re.MULTILINE)
    missing = [p for p in actual_params if p not in documented_params]
    hallucinated = [p for p in documented_params if p not in actual_params]
    return {
        "actual_params": actual_params,
        "documented_params": documented_params,
        "missing_params": missing,
        "hallucinated_params": hallucinated,
        "signature_match": not missing and not hallucinated,
    }
