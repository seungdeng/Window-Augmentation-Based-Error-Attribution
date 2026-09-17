import os
import json
import torch
import random
import re
from typing import Tuple, Dict, Any, Optional
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


# ===== Window-Based Final Judge (shared helper for the window-augmentation methods) =====
FINAL_WINDOW_RADIUS = 5  # ±5 → 총 11 스텝

def _render_segment(chat_history: list, start: int, end_exclusive: int, is_handcrafted: bool) -> str:
    idx_key = "role" if is_handcrafted else "name"
    seg = []
    for i in range(start, end_exclusive):
        entry = chat_history[i]
        agent = entry.get(idx_key, "Unknown Agent")
        content = entry.get("content", "")
        seg.append(f"Step {i} - {agent}: {content}")
    return "\n".join(seg)

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

