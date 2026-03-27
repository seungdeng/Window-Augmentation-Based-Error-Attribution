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

if __name__ == "__main__":
    main()