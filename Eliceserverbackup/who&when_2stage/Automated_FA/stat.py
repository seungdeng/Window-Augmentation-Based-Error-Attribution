# stat.py
import re
import sys
from collections import Counter
from pathlib import Path

# 확장된 패턴: Confidence score, Confidence, conf=
CONF_PATTERNS = [
    re.compile(r"Confidence\s*score\s*[:=]\s*([01](?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"\bConfidence\s*[:=]\s*([01](?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"\bconf\s*=\s*([01](?:\.\d+)?)", re.IGNORECASE),
]

def extract_scores(text: str):
    scores = []
    for pat in CONF_PATTERNS:
        for m in pat.finditer(text):
            try:
                v = float(m.group(1))
                if 0.0 <= v <= 1.0:
                    scores.append(v)
            except Exception:
                pass
    return scores

def bin_scores(scores):
    """
    0.0~0.9 구간을 0.1 단위 10개 버킷으로 집계.
    1.0은 0.9~1.0 버킷에 포함.
    """
    buckets = Counter()
    for s in scores:
        idx = int(s * 10)
        if idx >= 10:
            idx = 9
        buckets[idx] += 1
    return buckets

def print_distribution(buckets, total, verbose=False):
    print(f"Total scores found: {total}")
    if total == 0 and verbose:
        print("※ 추출된 스코어가 없습니다. 패턴/경로/인코딩을 확인하세요.")
    print()
    for i in range(10):
        lo = i / 10
        hi = (i + 1) / 10
        label = f"{lo:.1f} ~ {hi:.1f}"
        cnt = buckets.get(i, 0)
        pct = (cnt / total * 100) if total else 0.0
        print(f"{label}: {cnt:5d}  ({pct:5.1f}%)")

def read_text_from_path(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def main():
    if len(sys.argv) < 2:
        print("Usage: python stat.py <path_to_file_or_dir> [--verbose]")
        sys.exit(1)

    path_str = sys.argv[1]
    verbose = ("--verbose" in sys.argv[2:])

    path = Path(path_str)
    all_scores = []

    if path.is_file():
        if verbose:
            print(f"[INFO] Reading file: {path}")
        text = read_text_from_path(path)
        all_scores.extend(extract_scores(text))
    elif path.is_dir():
        if verbose:
            print(f"[INFO] Scanning directory: {path} (*.txt)")
        for fp in sorted(path.glob("*.txt")):
            if verbose:
                print(f"[INFO]  - {fp}")
            text = read_text_from_path(fp)
            all_scores.extend(extract_scores(text))
    else:
        print(f"[ERROR] Path not found: {path}")
        sys.exit(1)

    buckets = bin_scores(all_scores)
    print_distribution(buckets, len(all_scores), verbose=verbose)

if __name__ == "__main__":
    main()
