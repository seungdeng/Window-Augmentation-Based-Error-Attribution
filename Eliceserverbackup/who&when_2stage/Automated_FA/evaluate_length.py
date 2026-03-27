import re
import json
import os
import argparse


import os
import re

def read_predictions(eval_file: str, keep: str = "first"):
    """
    Read predictions from a Chunk-Parallel (proposed) evaluation log file.

    Expected block format (repeats):
      === Final Prediction for <name>.json ===
      Agent Name: <agent>
      Step Number: <int>
      Reason: ...

    Args:
        eval_file: path to the evaluation log
        keep: policy when duplicate filenames appear in the log:
              - "first": keep the first occurrence, ignore later ones
              - "last":  keep the last occurrence (default for many logs)

    Returns:
        dict: {
          "<name>.json": {"predicted_agent": str, "predicted_step": str},
          ...
        }
    """
    if not os.path.exists(eval_file):
        print(f"Error: Evaluation file not found at {eval_file}")
        return {}

    try:
        with open(eval_file, "r", encoding="utf-8") as f:
            data = f.read()
    except Exception as e:
        print(f"Error reading evaluation file {eval_file}: {e}")
        return {}

    # Normalize newlines just in case
    data = data.replace("\r\n", "\n")

    # Strictly match ONLY the proposed format blocks.
    # Capture groups:
    #   1: filename (e.g., 1.json)
    #   2: block content until next "=== Final Prediction ..." or end of file
    block_pat = re.compile(
        r"(?:^|\n)===\s*Final Prediction for\s+([^\n:]+?\.json)\s*===\s*"
        r"(.*?)"
        r"(?=(?:^|\n)===\s*Final Prediction for\s+[^\n:]+?\.json\s*===|\Z)",
        re.DOTALL | re.IGNORECASE

        # r"(?:^|\n)\s*Prediction for\s+([^\n:]+?\.json)\s*\s*"
        # r"(.*?)"
        # r"(?=(?:^|\n)\s*Prediction for\s+[^\n:]+?\.json\s*|\Z)",
        # re.DOTALL | re.IGNORECASE
    )

    # Inside each block, extract Agent/Step
    agent_pat = re.compile(r"Agent Name:\s*([\w_]+)", re.IGNORECASE)
    step_pat  = re.compile(r"Step Number:\s*(\d+)",   re.IGNORECASE)

    predictions = {}
    seen = set()
    parsed_count = 0
    dup_count = 0

    for m in block_pat.finditer(data):
        fname = m.group(1).strip()
        content = m.group(2).strip()

        agent_m = agent_pat.search(content)
        step_m  = step_pat.search(content)

        if not (agent_m and step_m):
            print(f"Warning: Could not parse Agent/Step for {fname} in {eval_file}")
            preview = content[:160].replace("\n", "\\n")
            print(f"  Preview: {preview}")
            continue

        agent = agent_m.group(1).strip()
        step  = step_m.group(1).strip()

        if fname in predictions:
            dup_count += 1
            if keep.lower() == "first":
                # keep existing, ignore this one
                continue
            elif keep.lower() == "last":
                # overwrite with the latest
                predictions[fname] = {
                    "predicted_agent": agent,
                    "predicted_step": step,
                }
            else:
                print(f"Warning: Unknown keep policy '{keep}'. Using 'first'.")
                continue
        else:
            predictions[fname] = {
                "predicted_agent": agent,
                "predicted_step": step,
            }
            seen.add(fname)
            parsed_count += 1

    print(f"--- Predictions Read from {eval_file} (Chunk-Parallel) ---")
    print(f"Successfully parsed predictions for {parsed_count} files.")
    if dup_count:
        print(f"Note: Detected {dup_count} duplicate filename block(s). keep='{keep}'.")
    print("===========================================================")

    return predictions

def read_actual_data(labeled_json):
    try:
        with open(labeled_json, 'r', encoding='utf-8') as file:
            data = json.load(file)
        mistake_agent = data.get('mistake_agent')
        mistake_step = data.get('mistake_step')
        if mistake_agent is not None and mistake_step is not None:
            return str(mistake_agent), str(mistake_step)
        else:
            print(f"Warning: 'mistake_agent' or 'mistake_step' key missing in {labeled_json}")
            return None, None
    except FileNotFoundError:
        print(f"Error: Actual data file not found during read: {labeled_json}")
        return None, None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {labeled_json}")
        return None, None
    except Exception as e:
        print(f"Error reading actual data from {labeled_json}: {e}")
        return None, None

def evaluate_accuracy(predictions, data_path, total_files):
    correct_agent = 0
    correct_step = 0
    files_evaluated = 0

    # 정답 파일 목록 수집 (수정)
    correct_agent_files = []
    correct_step_files = []

    if total_files == 0:
        print("Error: No JSON files found in the data path to evaluate against.")
        return 0.0, 0.0

    print(f"\n--- Starting Evaluation ---")
    print(f"Total reference JSON files found in {data_path}: {total_files}")
    print(f"Predictions available for {len(predictions)} files.")
    print("=======================================")

    #(수정) 함수 추가
    def _extract_num(fname: str):
        m = re.search(r'(\d+)\.json$', str(fname).strip(), re.IGNORECASE)
        return int(m.group(1)) if m else str(fname).strip()
    # 에이전트 이름 정규화(대소문자/공백/보조 태그 제거)
    def _canon_agent(s: str) -> str:
        s = str(s).strip().lower()
        s = re.sub(r"[\s_\-]+", " ", s).strip()
        # (thought)/(analysis)/(plan)/(reflection) 같은 꼬리표 제거
        s = re.sub(r'\s*\([^)]*\)\s*', '', s)
        s = re.sub(r"\s*(?:->|→)\s*.*$", "", s)
        # 표기 통일(선택) - 필요 없으면 제거해도 됨
        s = s.replace('web surfer', 'websurfer')\
             .replace('file surfer', 'filesurfer')\
             .replace('computer terminal', 'computerterminal')
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    # 안전한 int 변환
    def _to_int_safe(x):
        try:
            return int(str(x).strip())
        except Exception:
            return None

    for idx, pred in predictions.items():
        idx = idx.strip()  # 파일명 공백 제거
        labeled_file = os.path.join(data_path, idx)

        if os.path.exists(labeled_file):
            files_evaluated += 1
            actual_agent, actual_step = read_actual_data(labeled_file)

            if actual_agent is not None and actual_step is not None:
                pred_agent = pred.get('predicted_agent', '')
                pred_step  = pred.get('predicted_step', '')

                # 부분 일치 → 정확 비교로 수정
                agent_hit = (_canon_agent(actual_agent) == _canon_agent(pred_agent))

                a_step = _to_int_safe(actual_step)
                p_step = _to_int_safe(pred_step)
                step_hit = (a_step is not None and p_step is not None and a_step == p_step)

                if agent_hit:
                    correct_agent += 1
                    correct_agent_files.append(idx)
                if step_hit:
                    correct_step += 1
                    correct_step_files.append(idx)
            else:
                print(f"Skipping evaluation for {idx} due to issues reading actual data.")
        else:
            print(f"Warning: Labeled file not found for prediction key '{idx}': {labeled_file}")

    # 보기 좋게 번호만 출력
    agent_nums = sorted(_extract_num(f) for f in correct_agent_files)
    step_nums  = sorted(_extract_num(f) for f in correct_step_files)

    print("\n--- Evaluation Summary ---")
    print(f"Total reference files in data_path: {total_files}")
    print(f"Predictions parsed from eval file:  {len(predictions)}")
    print(f"Files evaluated (prediction found & actual data read): {files_evaluated}")
    #print(f"Correct Agent Predictions: {correct_agent}")
    #print(f"Correct Step Predictions:  {correct_step}")
    print(f"Correct Agent Predictions: {correct_agent}  -> files: {agent_nums}")
    print(f"Correct Step  Predictions: {correct_step}   -> files: {step_nums}")

    agent_accuracy = (correct_agent / total_files) * 100 if total_files > 0 else 0.0
    step_accuracy  = (correct_step  / total_files) * 100 if total_files > 0 else 0.0

    return agent_accuracy, step_accuracy
def _get_total_steps_from_json(labeled_json_path: str):
    """
    가능한 다양한 스키마를 대응하여 '총 스텝 수'를 추출.
    - 리스트 계열 키: messages, chat_history, history, dialogue, steps, log
    - 정수 계열 키: num_steps, total_steps
    없으면 None 반환(구간 집계에서 제외).
    """
    try:
        with open(labeled_json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return None

    # 1) 리스트 기반 길이
    for k in ["messages", "chat_history", "history", "dialogue", "steps", "log"]:
        v = d.get(k)
        if isinstance(v, list):
            return len(v)

    # 2) 정수 기반 길이
    for k in ["num_steps", "total_steps"]:
        v = d.get(k)
        try:
            n = int(v)
            if n > 0:
                return n
        except Exception:
            pass

    return None
def evaluate_by_step_ranges(
    predictions: dict,
    data_path: str,
    ranges: list[tuple[int, int]]
):
    """
    총 스텝 길이(total steps)가 특정 구간에 속하는 파일들에 대해
    구간별 데이터셋 개수 / Agent 정확도 / Step 정확도를 계산.

    ranges: [(51,70), (81,90), (91,110), (110,130)] 처럼 (lo, hi) 튜플 리스트.
            '닫힌 구간'으로 처리합니다: lo <= steps <= hi
            (요청 반영: 겹치는 경계값은 양쪽 구간 모두에 포함될 수 있음)

    반환값: 구간별 집계 사전
    """
    # 내부 유틸 (evaluate_accuracy()와 동일 로직)
    def _canon_agent(s: str) -> str:
        s = str(s or "").strip().lower()
        s = re.sub(r"[\s_\-]+", " ", s).strip()
        s = re.sub(r'\s*\([^)]*\)\s*', '', s)           # (thought) 등 꼬리표 제거
        s = re.sub(r"\s*(?:->|→)\s*.*$", "", s)         # 뒤 꼬리 잘라내기
        s = (s.replace('web surfer', 'websurfer')
               .replace('file surfer', 'filesurfer')
               .replace('computer terminal', 'computerterminal'))
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def _to_int_safe(x):
        try:
            return int(str(x).strip())
        except Exception:
            return None

    # 1) 데이터셋 전수: 각 파일의 총 스텝 수 미리 구해두기
    all_json_files = [f for f in os.listdir(data_path)
                      if f.endswith(".json") and os.path.isfile(os.path.join(data_path, f))]

    total_steps_map = {}   # fname -> total_steps or None
    for fname in all_json_files:
        total_steps_map[fname] = _get_total_steps_from_json(os.path.join(data_path, fname))

    # 2) 구간 준비
    # 구간 키는 문자열로 예쁘게
    pretty = lambda lo, hi: f"[{lo}–{hi}]"
    bins = {
        pretty(lo, hi): {
            "range": (lo, hi),
            "files": set(),       # 이 구간(데이터셋 개수)
            "agent_hits": 0,
            "step_hits": 0
        }
        for (lo, hi) in ranges
    }

    # 3) 파일을 구간에 배치
    for fname, steps in total_steps_map.items():
        if steps is None:
            continue  # 총 스텝 정보를 모르면 구간 집계에서 제외
        for (lo, hi) in ranges:
            if lo <= steps <= hi:
                bins[pretty(lo, hi)]["files"].add(fname)

    # 4) 예측과 정답 비교하여 구간별 정확도 집계
    for fname, pred in predictions.items():
        labeled_file = os.path.join(data_path, fname)
        if not os.path.exists(labeled_file):
            continue

        actual_agent, actual_step = read_actual_data(labeled_file)
        if actual_agent is None or actual_step is None:
            continue

        pred_agent = pred.get("predicted_agent", "")
        pred_step  = pred.get("predicted_step", "")

        agent_hit = (_canon_agent(actual_agent) == _canon_agent(pred_agent))
        a_step = _to_int_safe(actual_step)
        p_step = _to_int_safe(pred_step)
        step_hit = (a_step is not None and p_step is not None and a_step == p_step)

        # 이 파일이 속한 모든 구간에 정답 카운트를 올린다(겹치는 구간 허용).
        steps = total_steps_map.get(fname)
        if steps is None:
            continue
        for (lo, hi) in ranges:
            key = pretty(lo, hi)
            if fname in bins[key]["files"]:
                if agent_hit:
                    bins[key]["agent_hits"] += 1
                if step_hit:
                    bins[key]["step_hits"] += 1

    # 5) 결과 계산(분모=해당 구간의 데이터셋 개수)
    results = {}
    print("\n--- Accuracy by Step-Length Ranges ---")
    for key, info in bins.items():
        n = len(info["files"])  # 분모
        ah = info["agent_hits"]
        sh = info["step_hits"]
        agent_acc = (ah / n * 100.0) if n > 0 else 0.0
        step_acc  = (sh / n * 100.0) if n > 0 else 0.0

        results[key] = {
            "dataset_count": n,
            "agent_correct": ah,
            "step_correct": sh,
            "agent_accuracy": agent_acc,
            "step_accuracy": step_acc,
        }

        print(f"Range {key}:")
        print(f"  - Dataset Count: {n}")
        print(f"  - Agent  Acc: {agent_acc:.2f}%  (hits={ah}/{n})")
        print(f"  - Step   Acc: {step_acc:.2f}%  (hits={sh}/{n})")
    print("--------------------------------------\n")

    return results



def main():
    parser = argparse.ArgumentParser(description="Evaluate agent and step prediction accuracy from an evaluation log file.")
    parser.add_argument(
        "--data_path",
        type=str,
        default='../Who&When/Algorithm-Generated',
        help="Path to the directory containing the ground truth files."
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        required=True,
        help="Path to the evaluation log file containing the predictions."
    )
    # (옵션) 구간 커스터마이즈를 원하시면 인자 추가로 확장 가능
    args = parser.parse_args()

    data_path = args.data_path
    eval_file = args.eval_file

    if not os.path.isdir(data_path):
        print(f"Error: Data directory not found at {data_path}")
        actual_total_files = 0
    else:
        try:
            json_files_in_data_path = [
                f for f in os.listdir(data_path)
                if f.endswith('.json') and os.path.isfile(os.path.join(data_path, f))
            ]
            actual_total_files = len(json_files_in_data_path)
        except Exception as e:
            print(f"Error reading data directory {data_path}: {e}")
            actual_total_files = 0

    predictions = read_predictions(eval_file)

    agent_accuracy, step_accuracy = evaluate_accuracy(predictions, data_path, actual_total_files)

    print("\n--- Final Accuracy Results ---")
    print(f"Evaluation File: {eval_file}")
    print(f"Data Path:       {data_path}")
    print(f"Agent Accuracy: {agent_accuracy:.2f}%")
    print(f"Step Accuracy:  {step_accuracy:.2f}%")
    print(f"(Accuracy calculated based on {actual_total_files} total files in data path)")

    # ===== 추가: 구간별 집계 =====
    step_ranges = [(1,25),(26,50),(51,75),(76,100),(101,130)]
    _ = evaluate_by_step_ranges(predictions, data_path, step_ranges)


if __name__ == "__main__":
    main()
