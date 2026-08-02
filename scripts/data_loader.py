"""
Loads JSONL dataset files and converts to Hugging Face Dataset.
"""
import json
from typing import List, Dict

def load_jsonl(path: str) -> List[Dict[str, str]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

def save_jsonl(records: List[Dict[str, str]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def to_hf_dataset(records: List[Dict[str, str]]):
    try:
        from datasets import Dataset
    except ImportError as e:
        raise ImportError("The 'datasets' library is required. Install with: pip install datasets") from e
    return Dataset.from_list(records)
