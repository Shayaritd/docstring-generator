"""
ROUGE-L F1 computation for validation-set docstring quality.
"""
from typing import List, Dict


def _lcs_length(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def simple_rouge_l(reference: str, hypothesis: str) -> Dict[str, float]:
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not ref_tokens or not hyp_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    lcs = _lcs_length(ref_tokens, hyp_tokens)
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_rouge_l(references: List[str], hypotheses: List[str]) -> Dict[str, object]:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    if not references:
        raise ValueError("references/hypotheses cannot be empty")
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(references, hypotheses)]
        return {"rouge_l_f1": sum(scores) / len(scores), "method": "rouge_score_lib"}
    except ImportError:
        pass
    scores = [simple_rouge_l(r, h)["f1"] for r, h in zip(references, hypotheses)]
    return {"rouge_l_f1": sum(scores) / len(scores), "method": "simple_lcs_fallback"}
