import os
import json
import torch
import random
import re
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from transformers import pipeline as pipeline_function, AutoTokenizer, AutoModelForCausalLM, Pipeline
from tqdm import tqdm

def _get_sorted_json_files(directory_path):
    try:
        files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
        return sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    except FileNotFoundError:
        print(f"Error: Directory not found at {directory_path}")
        return []
    except Exception as e:
        print(f"Error reading or sorting files in {directory_path}: {e}")
        return []

def _load_json_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def _run_local_generation(model_obj, messages, model_family='llama', seed: Optional[int] = None):
    max_new_tokens=1024
    temperature=0.6
    top_p=0.95

    # ★ 시드 고정 (여기서만 처리; generate에 seed 인자 전달 금지)
    if seed is not None:
        import numpy as np, random, os
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    try:
        if model_family == 'llama' and isinstance(model_obj, Pipeline):
            pipe = model_obj
            terminators = [
                pipe.tokenizer.eos_token_id,
                pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            outputs = pipe(
                messages,
                max_new_tokens=max_new_tokens,
                eos_token_id=terminators,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=pipe.tokenizer.eos_token_id,
            )
            if outputs and outputs[0]["generated_text"] and isinstance(outputs[0]["generated_text"], list):
                 return outputs[0]["generated_text"][-1]["content"]
            else:
                 print("Warning: Unexpected output format from Llama pipeline.")
                 return None
        elif model_family == 'qwen' and isinstance(model_obj, tuple) and len(model_obj) == 2:
            model, tokenizer = model_obj
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=tokenizer.eos_token_id, # Use default EOS for Qwen generate
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response
        else:
            print(f"Error: Unsupported model_family '{model_family}' or incorrect model object type provided.")
            return None

    except Exception as e:
        print(f"Error during local model execution ({model_family}): {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_all_at_once_local(model_obj, directory_path: str, is_handcrafted: bool, model_family: str, seed: Optional[int] = None):
    print(f"\n--- Starting Local All-at-Once Analysis ({model_family}) ---")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if is_handcrafted else "name"

    for json_file in tqdm(json_files, desc=f"All-at-Once ({model_family})"):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "")

        if not chat_history:
            continue

        chat_content = "\n".join([
            f"{entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}" for entry in chat_history
        ])

        prompt = (
            "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
            f"The problem is:  {problem} \n"
            f"The Answer for the problem is: {ground_truth}\n"
            "Identify which agent made an error, at which step, and explain the reason for the error. "
            "Here's the conversation:\n\n" + chat_content +
            "\n\nBased on this conversation, please predict the following:\n"
            "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
            "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
            '{\n"agent a": "xx",\n"agent b": "xxxx",\n"agent c": "xxxxx",\n"agent a": "xxxxxxx"\n},\n'
            "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
            "3. The reason for your prediction."
            "Please answer in the format: Agent Name: (Your prediction)\n, Step Number: (Your prediction)\n, Reason for Mistake: (Your reason)\n."
        )

    
        system_prompt = "You are a helpful assistant skilled in analyzing conversations."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        assistant_response = _run_local_generation(model_obj, messages, model_family, seed=seed)

        print(f"Prediction for {json_file}:")
        if assistant_response:
            print(assistant_response)
        else:
            print("Failed to get prediction from local model.")
        print("\n" + "="*50 + "\n")

def analyze_step_by_step_local(model_obj, directory_path: str, is_handcrafted: bool, model_family: str, seed: Optional[int] = None):
    print(f"\n--- Starting Local Step-by-Step Analysis ({model_family}) ---")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if is_handcrafted else "name"

    for json_file in tqdm(json_files, desc=f"Step-by-Step ({model_family})"):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "")

        if not chat_history:
            continue

        current_conversation_history = ""
        error_found = False
        for idx, entry in enumerate(chat_history):
            agent_name = entry.get(index_agent, 'Unknown Agent')
            content = entry.get('content', '')
            current_conversation_history += f"Step {idx} - {agent_name}: {content}\n"

            prompt = (
                f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. The problem being addressed is: {problem}. "
                f"The Answer for the problem is: {ground_truth}\n"
                f"Here is the conversation history up to the current step:\n{current_conversation_history}\n"
                f"The most recent step ({idx}) was by '{agent_name}'.\n"
                "Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution. "
                "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
                "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process."
                "Attention: Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
            )

            system_prompt = "You are a helpful assistant skilled in analyzing conversations."

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            answer = _run_local_generation(model_obj, messages, model_family, seed)

            if not answer:
                print("Failed to get evaluation for this step from local model. Stopping analysis for this file.")
                error_found = True
                break

            if answer.lower().strip().startswith("1. yes"):
                print(f"\nPrediction for {json_file}: Error found.")
                print(f"Agent Name: {agent_name}")
                print(f"Step Number: {idx}")
                try:
                    reason = answer.split('Reason:', 1)[-1].strip()
                except:
                    reason = "[Could not extract reason]"
                print(f"Reason provided by LLM: {reason}")
                error_found = True
                break
            elif answer.lower().strip().startswith("1. no"):
                pass
            else:
                print(f"Warning: Unexpected response format from local LLM for step {idx} in {json_file}. Response: {answer[:100]}...")

        if not error_found:
            print(f"\nNo decisive errors found by step-by-step analysis in file {json_file}")

        print("\n" + "="*50 + "\n")


def _construct_binary_search_prompt_local(problem, answer, chat_segment_content, range_description, upper_half_desc, lower_half_desc):
     # Added answer back in based on previous logic, remove if not desired
    return (
        "You are an AI assistant tasked with analyzing a segment of a multi-agent conversation. Multiple agents are collaborating to address a user query, with the goal of resolving the query through their collective dialogue.\n"
        "Your primary task is to identify the location of the most critical mistake within the provided segment. Determine which half of the segment contains the single step where this crucial error occurs, ultimately leading to the failure in resolving the user’s query.\n"
        f"The problem to address is as follows: {problem}\n"
        f"The Answer for the problem is: {answer}\n"
        f"Review the following conversation segment {range_description}:\n\n{chat_segment_content}\n\n"
        f"Based on your analysis, predict whether the most critical error is more likely to be located in the upper half ({upper_half_desc}) or the lower half ({lower_half_desc}) of this segment.\n"
        "Please simply output either 'upper half' or 'lower half'. You should not output anything else."
    )

def _report_binary_search_error_local(chat_history, step, json_file, is_handcrafted):
    index_agent = "role" if is_handcrafted else "name"
    entry = chat_history[step]
    agent_name = entry.get(index_agent, 'Unknown Agent')

    print(f"\nPrediction for {json_file} (Binary Search Result):")
    print(f"Agent Name: {agent_name}")
    print(f"Step Number: {step}")
    print("\n" + "="*50 + "\n")

def _find_error_in_segment_local(model_obj, chat_history: list, problem: str, answer: str, start: int, end: int, json_file: str, is_handcrafted: bool, model_family: str):
    if start > end:
         print(f"Warning: Invalid range in binary search for {json_file} (start={start}, end={end}). Reporting last valid step.")
         _report_binary_search_error_local(chat_history, end if end >= 0 else 0, json_file, is_handcrafted)
         return
    if start == end:
        _report_binary_search_error_local(chat_history, start, json_file, is_handcrafted)
        return

    index_agent = "role" if is_handcrafted else "name"

    segment_history = chat_history[start : end + 1]
    if not segment_history:
        print(f"Warning: Empty segment in binary search for {json_file} (start={start}, end={end}). Reporting start index.")
        _report_binary_search_error_local(chat_history, start, json_file, is_handcrafted)
        return

    chat_content = "\n".join([
        f"{entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}"
        for entry in segment_history
    ])

    mid = start + (end - start) // 2

    range_description = f"from step {start} to step {end}"
    upper_half_desc = f"from step {start} to step {mid}"
    lower_half_desc = f"from step {mid + 1} to step {end}"

    prompt = _construct_binary_search_prompt_local(problem, answer, chat_content, range_description, upper_half_desc, lower_half_desc)

   
    system_prompt = "You are a helpful assistant skilled in analyzing conversations."


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    result = _run_local_generation(model_obj, messages, model_family)

    if not result:
        print(f"Model call failed for segment {start}-{end}. Stopping binary search for {json_file}.")
        return

    result_lower = result.lower().strip()

    if "upper half" in result_lower:
         _find_error_in_segment_local(model_obj, chat_history, problem, answer, start, mid, json_file, is_handcrafted, model_family)
    elif "lower half" in result_lower:
         new_start = min(mid + 1, end)
         _find_error_in_segment_local(model_obj, chat_history, problem, answer, new_start, end, json_file, is_handcrafted, model_family)
    else:
        print(f"Warning: Ambiguous response '{result}' from local LLM for segment {start}-{end}. Defaulting to upper half.")
        _find_error_in_segment_local(model_obj, chat_history, problem, answer, start, mid, json_file, is_handcrafted, model_family)


def analyze_binary_search_local(model_obj, directory_path: str, is_handcrafted: bool, model_family: str, seed: Optional[int] = None):
    print(f"\n--- Starting Local Binary Search Analysis ({model_family}) ---")
    json_files = _get_sorted_json_files(directory_path)

    for json_file in tqdm(json_files, desc=f"Binary Search ({model_family})"):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        answer = data.get("ground_truth", "")

        if not chat_history:
            continue

        _find_error_in_segment_local(
            model_obj=model_obj,
            chat_history=chat_history,
            problem=problem,
            answer=answer,
            start=0,
            end=len(chat_history) - 1,
            json_file=json_file,
            is_handcrafted=is_handcrafted,
            model_family=model_family
        )



# ===== Method 4: Chunk-Parallel (로컬 버전) =====
CHUNK_SIZE = 20
CHUNK_OVERLAP = 5
FINAL_WINDOW_RADIUS = 5  # ±5 → 총 11 스텝

def _make_chunks(L: int, C: int, O: int) -> List[Tuple[int, int]]:
    chunks = []
    if L <= 0 or C <= 0:
        return chunks
    s = 0
    while s < L:
        e = min(L, s + C)
        chunks.append((s, e))  # [s:e)
        if e == L:
            break
        s = max(0, e - O)
        if s >= L:
            break
        if s == 0 and e == L:
            break
    return chunks

def _render_segment(chat_history: list, start: int, end_exclusive: int, is_handcrafted: bool) -> str:
    idx_key = "role" if is_handcrafted else "name"
    seg = []
    for i in range(start, end_exclusive):
        entry = chat_history[i]
        agent = entry.get(idx_key, "Unknown Agent")
        content = entry.get("content", "")
        seg.append(f"Step {i} - {agent}: {content}")
    return "\n".join(seg)

def _parse_chunk_response(text: str) -> Tuple[Optional[str], Optional[int], Optional[float], str]:
    step = None
    conf = None
    reason = ""
    agent= None

    m = re.search(r"step\s*[:=]\s*(\d+)", text, flags=re.IGNORECASE)
    if m:
        try:
            step = int(m.group(1))
        except:
            step = None
    m = re.search(r"confidence\s*[:=]\s*([01](?:\.\d+)?)", text, flags=re.IGNORECASE)
    if m:
        try:
            conf = float(m.group(1))
        except:
            conf = None

    m = re.search(r"reason\s*[:=]\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        reason = m.group(1).strip()

    m = re.search(r"agent\s*[:=]\s*([^\r\n]+)", text, flags=re.IGNORECASE)
    if m: 
        agent = m.group(1).strip()

    return agent, step, conf, reason

def _judge_chunk_local(
    model_obj,
    model_family: str,
    problem: str,
    ground_truth: str,
    segment_text: str,
    chunk_range: Tuple[int, int],
    max_tokens: int = 1024,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    start, end_ex = chunk_range
    prompt = (
        "You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem.\n"
        f"Problem: {problem}\n"
        f"Answer: {ground_truth}\n"
        f"Consider ONLY the following conversation steps (global indices {start} to {end_ex-1}):\n\n"
        f"{segment_text}\n\n"
        "Select exactly ONE step most likely to be the earliest critical mistake.\n"
        "CALIBRATION RUBRIC for Confidence (strictly follow):\n"
        "- 0.0–0.19 = very weak evidence (speculative, multiple plausible alternatives)\n"
        "- 0.2–0.39 = weak evidence (some hints but contradictions/unknowns remain)\n"
        "- 0.4–0.59 = moderate (clear signals but with notable uncertainty)\n"
        "- 0.6–0.79 = strong (direct indicators, few plausible alternatives)\n"
        "- 0.8–1.0 = very strong (direct contradiction to goal/instructions with explicit traceable impact)\n"
        "If evidence is not explicit and traceable, keep confidence < 0.40.\n\n"
        "Respond ONLY in the format:\n"
        "Agent: <agent name as shown in the segment>\n"
        "Step: <global step integer>\n"
        "Reason: <brief, traceable reason citing concrete step content>\n"
        "Confidence: <number between 0 and 1 with two decimal place>\n"
    )
    messages = [
        {"role": "system", "content": "You are a precise, calibrated LLM judge for error localization."},
        {"role": "user", "content": prompt},
    ]
    resp = _run_local_generation(model_obj, messages, model_family, seed=seed) or ""
    agent, step, conf, reason = _parse_chunk_response(resp)
    if conf is not None:
        conf = max(0.0, min(1.0, conf))
    if step is not None and not (start <= step < end_ex):
        step = None
    return {"agent": agent, "step": step, "confidence": conf, "reason": reason, "raw": resp, "range": (start, end_ex)}


# ===== 헬퍼: 스테이지1/최종 응답 파서 =====
_stage1_agent_pat = re.compile(r"Agent\s*Name:\s*([^\n\r]+)", re.IGNORECASE)
_stage1_step_pat  = re.compile(r"Step\s*Number:\s*(\d+)", re.IGNORECASE)
_stage1_reason_pat= re.compile(r"Reason\s*for\s*Mistake:\s*([\s\S]+)$", re.IGNORECASE)

def _parse_stage1_response(text: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    1-stage 모델 출력에서 Agent/Step/Reason을 최대한 유연하게 추출.
    """
    if not text:
        return None, None, None
    agent = None
    step = None
    reason = None

    m = _stage1_agent_pat.search(text)
    if m:
        agent = m.group(1).strip()

    m = _stage1_step_pat.search(text)
    if m:
        try:
            step = int(m.group(1))
        except Exception:
            step = None

    m = _stage1_reason_pat.search(text)
    if m:
        reason = m.group(1).strip()

    return agent, step, reason


_final_step_pat  = re.compile(r"Step\s*:\s*(-?\d+)", re.IGNORECASE)
_final_reason_pat= re.compile(r"Reason\s*:\s*([\s\S]+)$", re.IGNORECASE)

def _parse_final_response(text: str) -> Tuple[Optional[int], str]:
    step = None
    reason = ""
    m = re.search(r"step\s*[:=]\s*(\d+)", text, flags=re.IGNORECASE)
    if m:
        try:
            step = int(m.group(1))
        except:
            step = None
    m = re.search(r"reason\s*[:=]\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        reason = m.group(1).strip()
    return step, reason

def _final_window_judge_local(
    model_obj,
    model_family: str,
    problem: str,
    ground_truth: str,
    chat_history: list,
    is_handcrafted: bool,
    pivot_step: int,
    max_tokens: int = 1024,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    L = len(chat_history)
    win_start = max(0, pivot_step - FINAL_WINDOW_RADIUS)
    win_end = min(L - 1, pivot_step + FINAL_WINDOW_RADIUS)
    window_text = _render_segment(chat_history, win_start, win_end + 1, is_handcrafted)

    prompt = (
        "You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem.\n"
        "Select exactly ONE global step index that best represents the true critical mistake.\n"
        "This should be the earliest step that directly leads to the failure.\n"
        f"Problem: {problem}\n"
        f"Answer: {ground_truth}\n\n"
        f"Window (global steps {win_start} to {win_end}):\n{window_text}\n\n"
        "Respond ONLY in the format:\n"
        "Step: <global step integer>\n"
        "Reason: <short reason>\n"
    )
    messages = [
        {"role": "system", "content": "You are a decisive final judge for pinpointing a single critical mistake step."},
        {"role": "user", "content": prompt},
    ]
    resp = _run_local_generation(model_obj, messages, model_family, seed=seed) or ""
    step, reason = _parse_final_response(resp)
    if step is None or not (win_start <= step <= win_end):
        step = pivot_step
    return {"step": step, "reason": reason, "raw": resp, "window": (win_start, win_end)} #"agent":agent,

def analyze_chunk_parallel_local(
    model_obj, directory_path: str, is_handcrafted: bool, model_family: str, seed: Optional[int] = None
):
    print(f"\n--- Starting Local Chunk-Parallel Analysis ({model_family}) ---")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if is_handcrafted else "name"

    for json_file in tqdm(json_files, desc=f"Chunk-Parallel ({model_family})"):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "")

        if not chat_history:
            print(f"Skipping {json_file}: No chat history found.")
            continue

        L = len(chat_history)
        chunks = _make_chunks(L, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            print(f"Skipping {json_file}: Cannot form chunks.")
            continue

        print(f"--- Analyzing File: {json_file} ---")
        per_chunk_results: List[Dict[str, Any]] = []
        for (s, e) in chunks:
            seg_text = _render_segment(chat_history, s, e, is_handcrafted)
            res = _judge_chunk_local(model_obj, model_family, problem, ground_truth, seg_text, (s, e), seed = seed)
            per_chunk_results.append(res)
            
            disp_step = res.get("step")
            disp_conf = res.get("confidence")
            # is_handcrafted 규칙에 따라 agent 키 선택
            idx_key = "role" if is_handcrafted else "name"
            if disp_step is not None:
                agent_name = chat_history[disp_step].get(idx_key, "Unknown Agent")
            else:
                agent_name = "N/A"

            print(
                f"Chunk [{s}:{e}) → agent={agent_name}, step={disp_step}, conf={disp_conf}, "
                f"reason={res.get('reason','')}"
            )
        best = None
        best_conf = -1.0
        best_step = None
        for r in per_chunk_results:
            step = r.get("step")
            conf = r.get("confidence")
            if step is None:
                continue
            conf_val = float(conf) if (conf is not None) else 0.0
            if (conf_val > best_conf) or (conf_val == best_conf and (best_step is None or step < best_step)):
                best = r
                best_conf = conf_val
                best_step = step

        if best is None or best.get("step") is None:
            print(f"Warning: No valid chunk decision for {json_file}. Falling back to mid step.")
            pivot = L // 2
        else:
            pivot = int(best["step"])

        print(f"\nTop-1 by confidence → pivot step = {pivot} (conf={best_conf})")

        final_res = _final_window_judge_local(model_obj, model_family, problem, ground_truth, chat_history, is_handcrafted, pivot, seed=seed)
        final_step = final_res["step"]
        entry = chat_history[final_step]
        agent_name = entry.get(index_agent, "Unknown Agent")

        print(f"\n=== Final Prediction for {json_file} ===")
        print(f"Agent Name: {agent_name}")
        print(f"Step Number: {final_step}")
        print(f"Reason: {final_res.get('reason','')}")
        print(f"Pivot (from chunks): {pivot}, Window: {final_res['window']}")
        print("=" * 50 + "\n")


def analyze_all_at_once_with_window_local(
    model_obj,
    directory_path: str,
    is_handcrafted: bool,
    model_family: str,
    seed: Optional[int] = None,
):
    """
    1) All-at-once로 에이전트/스텝/이유를 1차 예측
    2) 1차 예측의 step(없으면 중간값)을 pivot으로 삼아 윈도우 final judge 실행
    3) 최종 예측을 표준 포맷으로 출력
    """
    print(f"\n--- Starting Local All-at-Once(+Window) Analysis ({model_family}) ---")
    json_files = _get_sorted_json_files(directory_path)
    idx_key = "role" if is_handcrafted else "name"

    for json_file in tqdm(json_files, desc=f"All-at-Once+Window ({model_family})"):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "")

        if not chat_history:
            continue

        # === 1-stage 프롬프트 구성 ===
        chat_content = "\n".join([
            f"{entry.get(idx_key, 'Unknown Agent')}: {entry.get('content', '')}"
            for entry in chat_history
        ])

        prompt = (
            "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
            f"The problem is:  {problem} \n"
            f"The Answer for the problem is: {ground_truth}\n"
            "Identify which agent made an error, at which step, and explain the reason for the error. "
            "Here's the conversation:\n\n" + chat_content +
            "\n\nBased on this conversation, please predict the following:\n"
            "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
            "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
            '{\n"agent a": "xx",\n"agent b": "xxxx",\n"agent c": "xxxxx",\n"agent a": "xxxxxxx"\n},\n'
            "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
            "3. The reason for your prediction."
            "Please answer in the format: Agent Name: (Your prediction)\n, Step Number: (Your prediction)\n, Reason for Mistake: (Your reason)\n."
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant skilled in analyzing conversations."},
            {"role": "user", "content": prompt},
        ]

        # === 1-stage 실행 ===
        stage1_resp = _run_local_generation(model_obj, messages, model_family, seed = seed) or ""
        agent_1, step_1, reason_1 = _parse_stage1_response(stage1_resp)

        print(f"Prediction for {json_file}:")
        print(stage1_resp if stage1_resp else "Failed to get prediction from local model.")
        print("\n" + "="*50 + "\n")

        # === pivot 결정 (1-stage step 없으면 중간값) ===
        L = len(chat_history)
        pivot = step_1 if isinstance(step_1, int) and 0 <= step_1 < L else (L // 2)

        # === 최종 윈도우 판정 ===
        final_res = _final_window_judge_local(
            model_obj, model_family, problem, ground_truth,
            chat_history, is_handcrafted, pivot
        )
        final_step = final_res["step"]
        # 에이전트 이름은 최종 step 기준으로 가져오기
        if isinstance(final_step, int) and 0 <= final_step < L:
            agent_final = chat_history[final_step].get(idx_key, agent_1 or "Unknown Agent")
        else:
            agent_final = agent_1 or "Unknown Agent"

        # === 최종 결과 표준 출력 ===
        print(f"=== Final Prediction for {json_file} ===")
        print(f"Agent Name: {agent_final}")
        print(f"Step Number: {final_step if final_step is not None else pivot}")
        print(f"Reason: {final_res.get('reason','').strip() or (reason_1 or '').strip()}")
        print(f"Pivot (from stage1): {pivot}, Window: {final_res['window']}")
        print("=" * 50 + "\n")


def analyze_step_by_step_with_window_local(
    model_obj, directory_path: str, is_handcrafted: bool, model_family: str,seed: Optional[int] = None
):
    print(f"\n--- Starting Local Step-by-Step(+Window) Analysis ({model_family}) ---")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if is_handcrafted else "name"

    for json_file in tqdm(json_files, desc=f"Step-by-Step+Window ({model_family})"):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "")

        if not chat_history:
            continue

        L = len(chat_history)
        current_conversation_history = ""
        found = False
        pivot = None
        first_agent = None
        first_reason = ""

        for idx, entry in enumerate(chat_history):
            agent_name = entry.get(index_agent, 'Unknown Agent')
            content = entry.get('content', '')
            current_conversation_history += f"Step {idx} - {agent_name}: {content}\n"

            prompt = (
                f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. The problem being addressed is: {problem}. "
                f"The Answer for the problem is: {ground_truth}\n"
                f"Here is the conversation history up to the current step:\n{current_conversation_history}\n"
                f"The most recent step ({idx}) was by '{agent_name}'.\n"
                "Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution. "
                "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
                "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process."
                "Attention: Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
            )


            messages = [
                {"role": "system", "content": "You are a helpful assistant skilled in analyzing conversations."},
                {"role": "user", "content": prompt},
            ]
            answer = _run_local_generation(model_obj, messages, model_family,seed = seed)

            if not answer:
                print("Failed to get evaluation for this step from local model. Stopping analysis for this file.")
                found = True  # 중단
                break

            ans_norm = answer.strip().lower()
            if ans_norm.startswith("1. yes"):
                # 1-Stage: 최초 오류 스텝 확정
                reason = ""
                try:
                    reason = answer.split('Reason:', 1)[-1].strip()
                except Exception:
                    reason = "[Could not extract reason]"
                pivot = max(0, min(idx, L - 1))
                first_agent = agent_name
                first_reason = reason

                print(f"\nPrediction for {json_file}: Error found.")
                print(f"Agent Name: {first_agent}")
                print(f"Step Number: {pivot}")
                print(f"Reason provided by LLM: {first_reason}")

                # 2-Stage: 윈도우 최종 판정
                final_res = _final_window_judge_local(
                    model_obj=model_obj,
                    model_family=model_family,
                    problem=problem,
                    ground_truth=ground_truth,
                    chat_history=chat_history,
                    is_handcrafted=is_handcrafted,
                    pivot_step=pivot,
                    seed=seed
                )
                final_step = final_res["step"]
                # 에이전트는 최종 스텝 기준으로 다시 가져옴(안전 보정)
                if isinstance(final_step, int) and 0 <= final_step < L:
                    final_agent = chat_history[final_step].get(index_agent, first_agent or "Unknown Agent")
                else:
                    final_agent = first_agent or "Unknown Agent"

                print(f"\n=== Final Prediction for {json_file} ===")
                print(f"Agent Name: {final_agent}")
                print(f"Step Number: {final_step if final_step is not None else pivot}")
                print(f"Reason: {final_res.get('reason','').strip() or first_reason}")
                print(f"Pivot (from step-by-step): {pivot}, Window: {final_res['window']}")
                print("=" * 50 + "\n")

                found = True
                break

            elif ans_norm.startswith("1. no"):
                # 계속 진행
                pass
            else:
                print(f"Warning: Unexpected response format from local LLM for step {idx} in {json_file}. Response: {answer[:100]}...")

        if not found:
            print(f"\nNo decisive errors found by step-by-step analysis in file {json_file}")
            print("\n" + "="*50 + "\n")



# def analyze_chunk_parallel_local(
#     model_obj, directory_path: str, is_handcrafted: bool, model_family: str,seed: Optional[int] = None
# ):
#     print(f"\n--- Starting Local Chunk-Parallel Analysis ({model_family}) ---")
#     json_files = _get_sorted_json_files(directory_path)
#     index_agent = "role" if is_handcrafted else "name"

#     for json_file in tqdm(json_files, desc=f"Chunk-Parallel ({model_family})"):
#         file_path = os.path.join(directory_path, json_file)
#         data = _load_json_data(file_path)
#         if not data:
#             continue

#         chat_history = data.get("history", [])
#         problem = data.get("question", "")
#         ground_truth = data.get("ground_truth", "")

#         if not chat_history:
#             print(f"Skipping {json_file}: No chat history found.")
#             continue

#         L = len(chat_history)
#         chunks = _make_chunks(L, CHUNK_SIZE, CHUNK_OVERLAP)
#         if not chunks:
#             print(f"Skipping {json_file}: Cannot form chunks.")
#             continue

#         print(f"--- Analyzing File: {json_file} ---")
#         per_chunk_results: List[Dict[str, Any]] = []
#         for (s, e) in chunks:
#             seg_text = _render_segment(chat_history, s, e, is_handcrafted)
#             res = _judge_chunk_local(model_obj, model_family, problem, ground_truth, seg_text, (s, e))
#             per_chunk_results.append(res)
            
#             disp_step = res.get("step")
#             disp_conf = res.get("confidence")
#             # is_handcrafted 규칙에 따라 agent 키 선택
#             idx_key = "role" if is_handcrafted else "name"
#             if disp_step is not None:
#                 agent_name = chat_history[disp_step].get(idx_key, "Unknown Agent")
#             else:
#                 agent_name = "N/A"

#             print(
#                 f"Chunk [{s}:{e}) → agent={agent_name}, step={disp_step}, conf={disp_conf}, "
#                 f"reason={res.get('reason','')}"
#             )
#         best = None
#         best_conf = -1.0
#         best_step = None
#         for r in per_chunk_results:
#             step = r.get("step")
#             conf = r.get("confidence")
#             if step is None:
#                 continue
#             conf_val = float(conf) if (conf is not None) else 0.0
#             if (conf_val > best_conf) or (conf_val == best_conf and (best_step is None or step < best_step)):
#                 best = r
#                 best_conf = conf_val
#                 best_step = step

#         if best is None or best.get("step") is None:
#             print(f"Warning: No valid chunk decision for {json_file}. Falling back to mid step.")
#             pivot = L // 2
#         else:
#             pivot = int(best["step"])

#         print(f"\nTop-1 by confidence → pivot step = {pivot} (conf={best_conf})")

#         final_res = _final_window_judge_local(model_obj, model_family, problem, ground_truth, chat_history, is_handcrafted, pivot,seed=seed)
#         final_step = final_res["step"]
#         entry = chat_history[final_step]
#         agent_name = entry.get(index_agent, "Unknown Agent")

#         print(f"\n=== Final Prediction for {json_file} ===")
#         print(f"Agent Name: {agent_name}")
#         print(f"Step Number: {final_step}")
#         print(f"Reason: {final_res.get('reason','')}")
#         print(f"Pivot (from chunks): {pivot}, Window: {final_res['window']}")
#         print("=" * 50 + "\n")

