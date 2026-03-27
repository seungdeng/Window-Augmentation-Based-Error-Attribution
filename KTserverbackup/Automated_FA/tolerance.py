# filename: tolerance_accuracy.py
import re
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional

FINAL_BLOCK_RE = re.compile(
    r"===\s*Final\s*Prediction\s*for\s*(\d+)\.json\s*===\s*"
    r"(?:.*?\n)*?"                              # (에이전트, 이유 줄 등 스킵)
    r"Step\s*Number\s*:\s*([0-9]+)",            # 예측 Step Number
    flags=re.IGNORECASE
)

def parse_predictions(log_path: Path) -> Dict[int, int]:
    """
    txt 로그에서 최종 예측 Step Number를 파싱.
    반환: {sample_id(int): pred_step(int)}
    """
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    preds: Dict[int, int] = {}
    for m in FINAL_BLOCK_RE.finditer(text):
        sid = int(m.group(1))
        step = int(m.group(2))
        preds[sid] = step
    return preds

def extract_int(s: str) -> Optional[int]:
    """
    문자열에서 첫 번째 정수 추출 (예: '3', 'step 3', '03' 등).
    정수가 없으면 None.
    """
    m = re.search(r"-?\d+", str(s))
    return int(m.group()) if m else None

def load_ground_truth(json_dir: Path, start_id: int, end_id: int) -> Dict[int, int]:
    """
    Ground Truth JSON들의 'mistake_step'을 읽어 정수로 변환.
    반환: {sample_id(int): gt_step(int)}
    """
    gts: Dict[int, int] = {}
    for sid in range(start_id, end_id + 1):
        p = json_dir / f"{sid}.json"
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        raw = data.get("mistake_step", None)
        step = None
        if raw is not None:
            if isinstance(raw, int):
                step = raw
            else:
                step = extract_int(str(raw))
        if step is not None:
            gts[sid] = step
    return gts

def compute_accuracy(
    preds: Dict[int, int],
    gts: Dict[int, int],
    tolerances: List[int]
) -> List[Tuple[int, int, int, float]]:
    """
    각 오차범위에 대해 (tol, correct, total, accuracy) 리스트 반환.
    예측/정답 모두 존재하는 샘플만 평가.
    """
    ids = sorted(set(preds.keys()) & set(gts.keys()))
    total = len(ids)
    results = []
    for tol in tolerances:
        correct = 0
        for sid in ids:
            if abs(preds[sid] - gts[sid]) <= tol:
                correct += 1
        acc = (correct / total) if total > 0 else 0.0
        results.append((tol, correct, total, acc))
    return results

def main():
    ap = argparse.ArgumentParser(description="Tolerance accuracy (0~5) for step predictions.")
    ap.add_argument("--log", type=str, default="./outputs/[FINAL]step-by-step(GPT)/FINAL(WINDOW=3)/step_by_step_gpt-4o_alg_generated.txt", help="예측 로그 txt 파일 경로")
    ap.add_argument("--json_dir", type=str, default="../Who_and_When/Algorithm-Generated", help="GroundTruth JSON 디렉토리")
    ap.add_argument("--start", type=int, default=1, help="시작 id (기본 1)")
    ap.add_argument("--end", type=int, required=True, help="끝 id (예: 126)")
    args = ap.parse_args()

    log_path = Path(args.log)
    json_dir = Path(args.json_dir)

    preds = parse_predictions(log_path)
    gts = load_ground_truth(json_dir, args.start, args.end)

    # 공통 샘플 통계
    common_ids = sorted(set(preds) & set(gts))
    missing_pred = sorted(set(range(args.start, args.end+1)) - set(preds))
    missing_gt = sorted(set(range(args.start, args.end+1)) - set(gts))

    print("=== Coverage ===")
    print(f"Range             : {args.start}..{args.end} (total {args.end - args.start + 1})")
    print(f"Parsed predictions: {len(preds)} (used {len(common_ids)})")
    print(f"Found groundtruth : {len(gts)} (used {len(common_ids)})")
    if missing_pred:
        print(f"Missing prediction ids (ignored): {missing_pred[:10]}{' ...' if len(missing_pred)>10 else ''}")
    if missing_gt:
        print(f"Missing ground truth ids (ignored): {missing_gt[:10]}{' ...' if len(missing_gt)>10 else ''}")
    print()

    # 오차범위 0~5
    rows = compute_accuracy(preds, gts, tolerances=list(range(0, 6)))
    print("=== Accuracy by Tolerance ===")
    print("Tol | Correct / Total | Accuracy")
    for tol, correct, total, acc in rows:
        print(f"{tol:>3} | {correct:>7} / {total:<5} | {acc*100:6.2f}%")

if __name__ == "__main__":
    main()
