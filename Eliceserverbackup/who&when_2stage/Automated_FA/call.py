# filename: count_llm_calls.py
import re
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

NO_DECISIVE_RE = re.compile(
    r"No decisive errors found by step-by-step analysis in file\s+(\d+)\.json",
    flags=re.IGNORECASE
)
PREDICTION_RE = re.compile(
    r"Prediction\s+for\s+(\d+)\.json:\s*Error\s*found\.",
    flags=re.IGNORECASE
)
STEP_NUM_RE = re.compile(
    r"Step\s*Number\s*:\s*([0-9]+)",
    flags=re.IGNORECASE
)

def load_history_lengths(dataset_dir: Path) -> Dict[int, int]:
    """
    dataset_dir 안의 N.json 파일들을 읽어 {N: len(history)} 사전으로 반환.
    파일명이 '123.json' 처럼 정수형 이름이라고 가정.
    """
    history_len = {}
    for p in dataset_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".json":
            m = re.fullmatch(r"(\d+)\.json", p.name)
            if not m:
                continue
            idx = int(m.group(1))
            try:
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                h = data.get("history", [])
                if not isinstance(h, list):
                    # 혹시 형태가 다르면 0으로 처리
                    hlen = 0
                else:
                    hlen = len(h)
                history_len[idx] = hlen
            except Exception:
                # 깨진 파일 등은 0으로 처리
                history_len[idx] = 0
    return history_len

def parse_log_for_counts(log_path: Path) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    로그를 위에서부터 순회하며 두 종류의 정보를 수집:
      A) 'No decisive...' 라인의 파일 번호 목록 → history 길이를 더함
      B) 'Prediction for X.json: Error found.' 이후 가까운 곳의 'Step Number: k' → (X, k+1) 호출 합산
    반환:
      (no_decisive_ids, error_tuples)
        - no_decisive_ids: [파일번호, ...]
        - error_tuples: [(파일번호, step_number_plus_one), ...]
    """
    no_decisive_ids: List[int] = []
    error_tuples: List[Tuple[int, int]] = []

    with log_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]

        m_nd = NO_DECISIVE_RE.search(line)
        if m_nd:
            no_decisive_ids.append(int(m_nd.group(1)))
            i += 1
            continue

        m_pred = PREDICTION_RE.search(line)
        if m_pred:
            file_id = int(m_pred.group(1))
            # 다음 몇 줄 안에서 Step Number를 찾는다(블록 구분선 전까지 탐색)
            step_calls = None
            j = i + 1
            while j < len(lines):
                # 구분선 만나면 블록 종료로 간주
                if re.match(r"=+\s*$", lines[j]):
                    break
                m_step = STEP_NUM_RE.search(lines[j])
                if m_step:
                    step_num = int(m_step.group(1))
                    step_calls = step_num + 1  # 예시 규칙: 0 -> 1회
                    break
                j += 1
            if step_calls is None:
                # Step Number를 못 찾았으면 보수적으로 1회로 처리(필요시 0으로 바꿔도 됨)
                step_calls = 1
            error_tuples.append((file_id, step_calls))
            i = j + 1
            continue

        i += 1

    return no_decisive_ids, error_tuples

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_path", type=str, required=True, help="로그 txt 파일 경로")
    ap.add_argument("--dataset_dir", type=str, required=True, help="1.json~N.json이 있는 디렉터리 경로")
    args = ap.parse_args()

    log_path = Path(args.log_path)
    dataset_dir = Path(args.dataset_dir)

    history_len_map = load_history_lengths(dataset_dir)
    no_decisive_ids, error_tuples = parse_log_for_counts(log_path)

    total_calls = 0

    # A) No decisive… → 해당 파일의 history 길이만큼
    for fid in no_decisive_ids:
        hlen = history_len_map.get(fid, 0)
        total_calls += hlen

    # B) Prediction… Error found. → (step_number + 1)
    for fid, step_calls in error_tuples:
        total_calls += step_calls

    # 총 호출 횟수만 출력 (원하시면 상세 breakdown도 추가 가능)
    print(total_calls)

if __name__ == "__main__":
    main()
