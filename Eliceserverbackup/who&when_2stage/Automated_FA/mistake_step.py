# stat_mistake_step.py
import os
import json
from collections import Counter
from pathlib import Path

def load_json(fp: Path):
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read {fp}: {e}")
        return None

def main(directory: str):
    path = Path(directory)
    counts = Counter()


    for fp in sorted(path.glob("*.json")):
        data = load_json(fp)
        if not data:
            continue

        step = data.get("mistake_step")
        if step is None:
            continue
        try:
            step = int(step)
        except Exception:
            pass  # 문자열 그대로 둘 수도 있음
        counts[step] += 1

    print(f"Total files parsed: {sum(counts.values())}\n")
    print("=== mistake_step 분포 (오름차순) ===")
    for step, cnt in sorted(counts.items(), key=lambda x: int(x[0])):
        print(f"Step {step:>3}: {cnt}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python stat_mistake_step.py <json_directory>")
    else:
        main(sys.argv[1])
