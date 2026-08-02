"""
LLM-as-judge evaluation using Claude.
Requires: pip install anthropic, and ANTHROPIC_API_KEY environment variable.
"""
import json
import os
import time
from typing import Dict, List, Optional

JUDGE_SYSTEM_PROMPT = """You are an expert Python code reviewer evaluating auto-generated docstrings.
Score on four dimensions (1-5): accuracy, completeness, clarity, style.
Respond with ONLY a JSON object: {"accuracy": 1-5, "completeness": 1-5, "clarity": 1-5, "style": 1-5, "reason": "..."}
"""


def build_judge_prompt(code: str, generated_docstring: str) -> str:
    return f"Function code:\n```python\n{code}\n```\n\nGenerated docstring:\n\"\"\"\n{generated_docstring}\n\"\"\"\n\nEvaluate this docstring."


def _parse_judge_response(raw_text: str) -> Dict:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    return json.loads(cleaned)


def judge_single_example(code: str, generated_docstring: str, client=None, model: str = "claude-sonnet-4-6", max_retries: int = 3) -> Dict:
    try:
        import anthropic
    except ImportError as e:
        raise ImportError("Install with: pip install anthropic") from e
    if client is None:
        client = anthropic.Anthropic()
    prompt = build_judge_prompt(code, generated_docstring)
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model, max_tokens=300, system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            raw_text = response.content[0].text
            scores = _parse_judge_response(raw_text)
            for key in ("accuracy", "completeness", "clarity", "style"):
                if key not in scores or not isinstance(scores[key], (int, float)) or not (1 <= scores[key] <= 5):
                    raise ValueError(f"Invalid '{key}' score: {scores.get(key)}")
            return scores
        except json.JSONDecodeError as e:
            last_error = f"Judge response wasn't valid JSON: {e}"
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
    return {"accuracy": None, "completeness": None, "clarity": None, "style": None,
            "reason": f"[judge failed: {last_error}]", "error": last_error}


def judge_batch(examples: List[Dict], model: str = "claude-sonnet-4-6", rate_limit_delay: float = 0.5) -> List[Dict]:
    import anthropic
    client = anthropic.Anthropic()
    results = []
    for i, ex in enumerate(examples):
        print(f"  Judging example {i + 1}/{len(examples)}...")
        result = judge_single_example(ex["input"], ex["generated"], client=client, model=model)
        results.append(result)
        time.sleep(rate_limit_delay)
    return results
