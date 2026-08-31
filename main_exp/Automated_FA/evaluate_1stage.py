import re
import json
import os
import argparse
from typing import Dict, Tuple, Optional, List, Set

# -------------------------------
# Utilities
# -------------------------------

def _canon_agent(s: str) -> str:
    """에이전트 이름 정규화: 소문자, 공백을 하나로, 괄호 꼬리표 제거, _, - 제거."""
    s = str(s or "").strip().lower()
    s = re.sub(r"\s*\([^)]*\)\s*", "", s)  # (thought), (analysis) 등 제거
    s = re.sub(r"[\s_\-]+", " ", s)         # space/underscore/hyphen 통일
    s = s.strip()
    return s

def _to_int_safe(x) -> Optional[int]:
    try:
        return int(str(x).strip())
    except Exception:
        return None

def _extract_num(fname: str):
    m = re.search(r'(\d+)\.json$', str(fname).strip(), re.IGNORECASE)
    return int(m.group(1)) if m else str(fname).strip()

# -------------------------------
# Ground-truth reader & evaluator
# -------------------------------

def read_actual_data(labeled_json: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        with open(labeled_json, 'r', encoding='utf-8') as file:
            data = json.load(file)
        mistake_agent = data.get('mistake_agent')
        mistake_step  = data.get('mistake_step')
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


def _get_role_at_step(labeled_json_path: str, step: int) -> str:
    """
    해당 파일의 history에서 (1-based) step에 대응되는 턴의 role을 반환.
    범위 밖/형식 오류 시 "" 반환.
    """
    try:
        with open(labeled_json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        hist = data.get("history")
        if not isinstance(hist, list):
            return ""
        idx = step -1 # 1-based → 0-based
        if 0 <= idx < len(hist):
            role = (hist[idx] or {}).get("role", "")
            return _canon_agent(role)
        return ""
    except Exception as e:
        print(f"[WARN] Failed to map step to role for {labeled_json_path}: {e}")
        return ""


##디버깅
def _get_role_from_history(path: str, step: int) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        hist = d.get("history", [])
        i = (int(step) if step is not None else 1) 
        if 0 <= i < len(hist):
            return _canon_agent((hist[i] or {}).get("role", ""))
        return ""
    except Exception:
        return ""



def evaluate_accuracy(predictions: Dict[str, Dict[str, str]], data_path: str, total_files: int):
    correct_agent = 0
    correct_step  = 0
    files_evaluated = 0

    correct_agent_files: List[str] = []
    correct_step_files:  List[str] = []

    if total_files == 0:
        print("Error: No JSON files found in the data path to evaluate against.")
        return 0.0, 0.0

    print(f"\n--- Starting Evaluation ---")
    print(f"Total reference JSON files found in {data_path}: {total_files}")
    print(f"Predictions available for {len(predictions)} files.")
    print("=======================================")

    for idx, pred in predictions.items():
        idx = idx.strip()
        labeled_file = os.path.join(data_path, idx)


        if os.path.exists(labeled_file):
            files_evaluated += 1
            actual_agent, actual_step = read_actual_data(labeled_file)

            ##디버깅
            # gt_role_at_gt_step = _get_role_from_history(labeled_file, actual_step)
            # if _canon_agent(actual_agent) != gt_role_at_gt_step:
            #     print(f"[LABEL-MISMATCH] {idx}: label.mistake_agent={_canon_agent(actual_agent)!r} "
            #         f"history[gt_step-1].role={gt_role_at_gt_step!r}  gt_step={actual_step}")

            if actual_agent is not None and actual_step is not None:
                pred_agent = pred.get('predicted_agent', '')
                pred_step  = pred.get('predicted_step', '')

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

    agent_nums = sorted(_extract_num(f) for f in correct_agent_files)
    step_nums  = sorted(_extract_num(f) for f in correct_step_files)

    print("\n--- Evaluation Summary ---")
    print(f"Total reference files in data_path: {total_files}")
    print(f"Predictions parsed:                {len(predictions)}")
    print(f"Files evaluated:                   {files_evaluated}")
    print(f"Correct Agent Predictions: {correct_agent}  -> files: {agent_nums}")
    print(f"Correct Step  Predictions: {correct_step}   -> files: {step_nums}")

    agent_accuracy = (correct_agent / total_files) * 100 if total_files > 0 else 0.0
    step_accuracy  = (correct_step  / total_files) * 100 if total_files > 0 else 0.0
    return agent_accuracy, step_accuracy

# -------------------------------
# Predictions from Chunk-Top1 (step → role)
# -------------------------------

def read_predictions_chunk_top1(eval_file: str, data_path: str) -> Dict[str, Dict[str, str]]:
    """
    각 '--- Analyzing File: <name>.json ---' 블록에서
    Chunk 라인들을 파싱해 (conf 최대, 동률이면 start 최소) 후보 1개 선택.
    선택된 chunk의 step을 사용하여 data_path/<name>.json 의 history[step-1].role을
    예측 에이전트로 사용한다.

    Returns:
        { "<name>.json": {"predicted_agent": str, "predicted_step": str}, ... }
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

    data = data.replace("\r\n", "\n")

    analyzing_block_pat = re.compile(
        r"(?:^|\n)---\s*Analyzing File:\s*([^\n]+?\.json)\s*---\s*\n"
        r"(.*?)"
        r"(?=(?:^|\n)---\s*Analyzing File:|\Z)",
        re.DOTALL | re.IGNORECASE
    )

    # --- 청크 라인 패턴 (현재 로그 형식 호환) ---
    #  - '→' 또는 '->' 지원
    #  - agent=... , step=..., conf=... , reason=... (conf는 같은 줄 또는 다음 줄 "Confidence: x.x")
    chunk_block_pat = re.compile(
        r"Chunk\s*\[(\d+):(\d+)\)\s*(?:→|->)\s*"
        r"(?:(?:agent\s*=\s*([^,]+)\s*,\s*)?)"     # G3: optional agent (앞쪽)
        r"step\s*=\s*(\d+)\s*,\s*"                 # G4: step
        r"(?:(?:agent\s*=\s*([^,]+)\s*,\s*)?)"     # G5: optional agent (뒤쪽)
        r"(?:(?:conf\s*=\s*([0-9]*\.?[0-9]+)\s*,\s*)?)"  # G6: conf= (같은 줄)
        r"reason\s*=\s*(.*?)(?:\nConfidence:\s*([0-9]*\.?[0-9]+))?"  # G7: reason, G8: 다음 줄 Confidence:
        r"(?=\nChunk|\Z)",
        re.IGNORECASE | re.DOTALL
    )
    predictions: Dict[str, Dict[str, str]] = {}
    parsed_blocks = 0
    files_with_chunks = 0
    
    for m in analyzing_block_pat.finditer(data):
        fname = m.group(1).strip()
        body  = m.group(2)
        parsed_blocks += 1

        candidates = []
        for cm in chunk_block_pat.finditer(body):
            start = int(cm.group(1))
            step  = int(cm.group(4))
            # agent는 앞/뒤 어느 그룹에 있어도 우선 사용
            agent_model = (cm.group(3) or cm.group(5) or "").strip()

            # conf는 같은 줄의 conf=... 우선, 없으면 다음 줄 "Confidence: ..."
            conf_txt = (cm.group(6) or cm.group(8) or "").strip()
            try:
                conf = float(conf_txt) if conf_txt else 0.0
            except ValueError:
                conf = 0.0

            candidates.append({
                "conf": conf,
                "start": start,
                "step": step,
                "agent_model": agent_model,
            })

        if not candidates:
            continue

        files_with_chunks += 1
        # 정렬: conf 내림차순, start 오름차순
        candidates.sort(key=lambda t: (-t["conf"], t["start"]))
        best = candidates[0]

        pred_step = best["step"]
        pred_agent_from_log = best.get("agent_model", "").strip()  # ★ 1-stage: 로그 그대로

        predictions[fname] = {
            "predicted_agent": pred_agent_from_log,  # history 매핑 없이 모델 출력 그대로
            "predicted_step": str(pred_step),
        }

    print(f"--- Predictions Read from {eval_file} (Chunk-Top1 using agent from log) ---")
    print(f"Parsed 'Analyzing File' blocks: {parsed_blocks}")
    print(f"Blocks with chunk lines:        {files_with_chunks}")
    print(f"Files with predictions:         {len(predictions)}")
    print("===================================================")
    return predictions

def read_predictions_all_at_once(eval_file: str) -> Dict[str, Dict[str, str]]:
    """
    Parse logs like:
      === Final Prediction for 1.json ===
      Agent Name: WebSurfer
      Step Number: 4
      Reason: ...
      Pivot (from stage1): 2, Window: (0, 7)
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

    data = data.replace("\r\n", "\n")

    # === Final Prediction for <name>.json === 블록 단위
    block_pat = re.compile(
        r"Prediction for\s*([^\n]+?\.json)\s*:\s*\n(.*?)(?=(?:^|\n)===\s*Final Prediction for|\Z)",
        re.DOTALL | re.IGNORECASE
    )
    agent_pat = re.compile(r"Agent\s*Name\s*:\s*([^\r\n]+)", re.IGNORECASE)
    step_pat  = re.compile(r"Step\s*Number\s*:\s*(\d+)",   re.IGNORECASE)

    predictions: Dict[str, Dict[str, str]] = {}
    blocks = 0
    for m in block_pat.finditer(data):
        fname = m.group(1).strip()
        body  = m.group(2)

        a = agent_pat.search(body)
        s = step_pat.search(body)
        if not s:
            continue  # 스텝이 없으면 스킵

        pred_agent = (a.group(1).strip() if a else "").strip()
        pred_step  = s.group(1).strip()

        predictions[fname] = {
            "predicted_agent": pred_agent,
            "predicted_step":  pred_step,
        }
        blocks += 1

    print(f"--- Predictions Read from {eval_file} (All-at-once+Window) ---")
    print(f"Final-prediction blocks parsed: {blocks}")
    print(f"Files with predictions:         {len(predictions)}")
    print("===================================================")
    return predictions

def read_predictions_step_by_step(eval_file: str) -> Dict[str, Dict[str, str]]:
    """
    Parse logs like (step-by-step WITHOUT window):
      Prediction for 1.json: Error found.
      Agent Name: WebSurfer
      Step Number: 12
      Reason provided by LLM: ...

    We only need Agent/Step.
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

    data = data.replace("\r\n", "\n")

    # 블록은 "Prediction for <name>.json:" 로 시작해서 다음 "Prediction for" 또는 파일 끝까지
    block_pat = re.compile(
        r"(?:^|\n)Prediction\s+for\s+([^\n]+?\.json)\s*:\s*.*?\n(.*?)(?=(?:^|\n)Prediction\s+for\s+|\Z)",
        re.DOTALL | re.IGNORECASE
    )
    agent_pat = re.compile(r"Agent\s*Name\s*:\s*([^\r\n]+)", re.IGNORECASE)
    step_pat  = re.compile(r"Step\s*Number\s*:\s*(\d+)",   re.IGNORECASE)

    predictions: Dict[str, Dict[str, str]] = {}
    blocks = 0
    for m in block_pat.finditer(data):
        fname = m.group(1).strip()
        body  = m.group(2)

        a = agent_pat.search(body)
        s = step_pat.search(body)
        if not s:
            # 스텝이 없으면 스킵
            continue

        pred_agent = (a.group(1).strip() if a else "").strip()
        pred_step  = s.group(1).strip()

        predictions[fname] = {
            "predicted_agent": pred_agent,
            "predicted_step":  pred_step,
        }
        blocks += 1

    print(f"--- Predictions Read from {eval_file} (Step-by-step, no window) ---")
    print(f"Simple step-by-step blocks parsed: {blocks}")
    print(f"Files with predictions:           {len(predictions)}")
    print("===================================================")
    return predictions
# -------------------------------
# CLI
# -------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate agent & step accuracy using Chunk-Top1; agent = history[step-1].role (no reason-text parsing)."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the directory containing the ground truth JSON files (also used to read history for role)."
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        required=True,
        help="Path to the evaluation log file containing the chunk lines."
    )
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

    # 1) 먼저 청크 로그 포맷 시도
    predictions = read_predictions_chunk_top1(eval_file, data_path)
    pred_source = "Chunk-Top1"

    # 2) 청크 포맷이 0건이면 All-at-once(+Window) 포맷 시도
    if len(predictions) == 0:
        predictions = read_predictions_all_at_once(eval_file)
        pred_source = "All-at-once(+Window)"
    if len(predictions) == 0:
        predictions = read_predictions_step_by_step(eval_file)
        pred_source = "Step-by-step(+window)"
    # 3) 평가
    agent_acc, step_acc = evaluate_accuracy(predictions, data_path, actual_total_files)

    print(f"\n--- Final Accuracy Results ({pred_source}) ---")


    print("\n--- Final Accuracy Results (Chunk-Top1 via history.role) ---")
    print(f"Evaluation File: {eval_file}")
    print(f"Data Path:       {data_path}")
    print(f"Agent Accuracy: {agent_acc:.2f}%")
    print(f"Step Accuracy:  {step_acc:.2f}%")
    print(f"(Accuracy calculated based on {actual_total_files} total files in data path)")

    # 누락 예측 디버그 출력
    data_files = [f for f in os.listdir(data_path) if f.endswith(".json")]
    def _canon_fname(s: str) -> str:
        m = re.search(r'(\d+)\.json$', s.strip(), re.IGNORECASE)
        return f"{int(m.group(1))}.json" if m else s.strip()

    label_set = {_canon_fname(f) for f in data_files}
    pred_set  = {_canon_fname(k) for k in predictions.keys()}
    missing   = sorted(
        (label_set - pred_set),
        key=lambda x: int(re.search(r'(\d+)', x).group(1)) if re.search(r'(\d+)', x) else 10**9
    )
    if missing:
        head = missing[:20]
        tail = len(missing) - 20
        print("[DEBUG] Missing predictions for files:", head, ("...(+%d more)" % tail if tail > 0 else ""))

if __name__ == "__main__":
    main()