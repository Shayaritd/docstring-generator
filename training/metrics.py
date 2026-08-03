"""
BLEU score computation for validation-set docstring quality.

Prefers `sacrebleu` (standard, citable BLEU implementation) if installed,
then falls back to a self-contained corpus BLEU implementation so
evaluation never silently fails due to a missing optional dependency.
"""

import math
from collections import Counter
from typing import List, Dict


def _get_ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def simple_corpus_bleu(references: List[str], hypotheses: List[str], max_n: int = 4, smooth: bool = True) -> float:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    if not references:
        raise ValueError("references/hypotheses cannot be empty")

    ref_tokens_list = [r.split() for r in references]
    hyp_tokens_list = [h.split() for h in hypotheses]

    total_hyp_len = 0
    total_ref_len = 0
    clipped_counts = [0] * max_n
    total_counts = [0] * max_n

    for ref_tokens, hyp_tokens in zip(ref_tokens_list, hyp_tokens_list):
        total_hyp_len += len(hyp_tokens)
        total_ref_len += len(ref_tokens)
        for n in range(1, max_n + 1):
            hyp_ngrams = _get_ngrams(hyp_tokens, n)
            ref_ngrams = _get_ngrams(ref_tokens, n)
            clipped = sum(min(count, ref_ngrams.get(ng, 0)) for ng, count in hyp_ngrams.items())
            clipped_counts[n - 1] += clipped
            total_counts[n - 1] += max(len(hyp_tokens) - n + 1, 0)

    if any(c == 0 for c in total_counts):
        return 0.0

    if smooth:
        epsilon = 1e-7
        precisions = [
            (clipped_counts[i] if clipped_counts[i] > 0 else epsilon) / total_counts[i]
            for i in range(max_n)
        ]
    else:
        if any(c == 0 for c in clipped_counts):
            return 0.0
        precisions = [clipped_counts[i] / total_counts[i] for i in range(max_n)]

    log_precision_avg = sum(math.log(p) for p in precisions) / max_n
    brevity_penalty = 1.0 if total_hyp_len >= total_ref_len else math.exp(1 - total_ref_len / max(total_hyp_len, 1))

    return brevity_penalty * math.exp(log_precision_avg) * 100


def compute_bleu(references: List[str], hypotheses: List[str]) -> Dict[str, object]:
    try:
        import sacrebleu
        score = sacrebleu.corpus_bleu(hypotheses, [references]).score
        return {"bleu": score, "method": "sacrebleu"}
    except ImportError:
        pass

    try:
        import evaluate
        bleu_metric = evaluate.load("bleu")
        result = bleu_metric.compute(predictions=hypotheses, references=[[r] for r in references])
        return {"bleu": result["bleu"] * 100, "method": "hf_evaluate"}
    except ImportError:
        pass

    score = simple_corpus_bleu(references, hypotheses)
    return {"bleu": score, "method": "simple_corpus_bleu_fallback"}
