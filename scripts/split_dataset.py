"""
Stratified 80/10/10 train/validation/test split.
"""
from typing import List, Dict, Tuple
from collections import Counter
from sklearn.model_selection import train_test_split
from features import extract_features

def _get_buckets(records: List[Dict[str, str]]) -> List[str]:
    return [extract_features(r["input"], r["output"]).complexity_bucket for r in records]

def stratified_split(records: List[Dict[str, str]], train_size: float = 0.8, val_size: float = 0.1,
                     test_size: float = 0.1, random_state: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    if abs((train_size + val_size + test_size) - 1.0) > 1e-6:
        raise ValueError("train_size + val_size + test_size must sum to 1.0")
    buckets = _get_buckets(records)
    bucket_counts = Counter(buckets)
    if any(count < 2 for count in bucket_counts.values()):
        raise ValueError(f"Cannot stratify: some buckets have <2 examples")
    indices = list(range(len(records)))
    train_idx, temp_idx, train_b, temp_b = train_test_split(
        indices, buckets, train_size=train_size, stratify=buckets, random_state=random_state
    )
    relative_val_size = val_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=relative_val_size, stratify=temp_b, random_state=random_state
    )
    return [records[i] for i in train_idx], [records[i] for i in val_idx], [records[i] for i in test_idx]

def split_summary(train: List[Dict], val: List[Dict], test: List[Dict]) -> Dict:
    return {"train": {"count": len(train), "buckets": dict(Counter(_get_buckets(train)))},
            "val": {"count": len(val), "buckets": dict(Counter(_get_buckets(val)))},
            "test": {"count": len(test), "buckets": dict(Counter(_get_buckets(test)))}}
