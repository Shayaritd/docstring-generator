"""
Validates a JSONL dataset file against the DocstringExample schema.
"""
import json
import sys
from pydantic import ValidationError
from schema import DocstringExample

def validate_file(path: str) -> None:
    total = 0
    valid = 0
    errors = []
    non_google_style = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append((line_num, f"Invalid JSON: {e}"))
                continue
            try:
                example = DocstringExample(**raw)
                valid += 1
                if not example.is_google_style():
                    non_google_style.append(line_num)
            except ValidationError as e:
                errors.append((line_num, str(e)))
    print(f"Total lines checked: {total}")
    print(f"Valid entries:       {valid}")
    print(f"Invalid entries:     {len(errors)}")
    if non_google_style:
        print(f"\nWarning: {len(non_google_style)} entries lack Google-style section headers")
    if errors:
        print("\nErrors:")
        for line_num, msg in errors:
            print(f"  Line {line_num}: {msg}")
        sys.exit(1)
    else:
        print("\nAll entries passed schema validation.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_dataset.py <path_to_jsonl>")
        sys.exit(1)
    validate_file(sys.argv[1])
