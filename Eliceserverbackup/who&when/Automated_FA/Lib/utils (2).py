import os
import json
import random
from openai import AzureOpenAI
from tqdm import tqdm
# --- Helper Functions ---

def _get_sorted_json_files(directory_path):
    """Gets and sorts JSON files numerically from a directory."""
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
    """Loads data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def _make_api_call(client, model, messages, max_tokens):
    """Makes an API call to Azure OpenAI."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return None

# --- All-at-Once Method ---

def all_at_once(client: AzureOpenAI, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int):
    """
    Analyzes chat history by feeding the entire conversation at once to the model.
    """
    print("\n--- Starting All-at-Once Analysis ---\n")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if is_handcrafted else "name"

    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "") # Keep ground truth if needed for evaluation

        if not chat_history:
            print(f"Skipping {json_file}: No chat history found.")
            continue

        chat_content = "\n".join([
            f"{entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}" for entry in chat_history
        ])

        prompt = (
            "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
            f"The problem is:  {problem}\n"
            f"The Answer for the problem is: {ground_truth}\n" # Included as per original code - remove if ground truth shouldn't be used in prompt
            "Identify which agent made an error, at which step, and explain the reason for the error. "
            "Here's the conversation:\n\n" + chat_content +
            "\n\nBased on this conversation, please predict the following:\n"
            "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
            "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
            """
            {
                "agent a": "xx",
                "agent b": "xxxx",
                "agent c": "xxxxx",
                "agent a": "xxxxxxx"
            },
            """
            "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
            "3. The reason for your prediction."
            "Please answer in the format: Agent Name: (Your prediction)\n Step Number: (Your prediction)\n Reason for Mistake: \n"
        )

        messages=[
            {"role": "system", "content": "You are a helpful assistant skilled in analyzing conversations."},
            {"role": "user", "content": prompt},
        ]

        result = _make_api_call(client, model, messages, max_tokens)

        print(f"Prediction for {json_file}:")
        if result:
            print(result)
        else:
            print("Failed to get prediction.")
        print("\n" + "="*50 + "\n")



# --- Step-by-Step Method ---

def step_by_step(client: AzureOpenAI, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int):
    """
    Analyzes chat history step by step, asking the model at each step if an error occurred.
    """
    print("\n--- Starting Step-by-Step Analysis ---\n")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if is_handcrafted else "name"

    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "") # Keep ground truth if needed

        if not chat_history:
            print(f"Skipping {json_file}: No chat history found.")
            continue

        print(f"--- Analyzing File: {json_file} ---")
        current_conversation_history = ""
        error_found = False
        for idx, entry in enumerate(chat_history):
            agent_name = entry.get(index_agent, 'Unknown Agent')
            content = entry.get('content', '')
            current_conversation_history += f"Step {idx} - {agent_name}: {content}\n"

            prompt = (
                f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. The problem being addressed is: {problem}. "
                f"The Answer for the problem is: {ground_truth}\n" # Included as per original code - remove if ground truth shouldn't be used
                f"Here is the conversation history up to the current step:\n{current_conversation_history}\n"
                f"The most recent step ({idx}) was by '{agent_name}'.\n"
                "Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution. "
                "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
                "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process."
                "Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
            )

            messages=[
                {"role": "system", "content": "You are a precise step-by-step conversation evaluator."},
                {"role": "user", "content": prompt},
            ]

            print(f"Evaluating Step {idx} by {agent_name}...")
            answer = _make_api_call(client, model, messages, max_tokens)

            if not answer:
                print("Failed to get evaluation for this step. Stopping analysis for this file.")
                error_found = True # Treat API error as unable to proceed
                break

            print(f"LLM Evaluation: {answer}")

            # Basic check for "Yes" at the beginning of the response
            if answer.lower().strip().startswith("1. yes"):
                print(f"\nPrediction for {json_file}: Error found.")
                print(f"Agent Name: {agent_name}")
                print(f"Step Number: {idx}")
                print(f"Reason provided by LLM: {answer.split('Reason:', 1)[-1].strip()}")
                error_found = True
                break # Stop processing this file once an error is found
            elif answer.lower().strip().startswith("1. no"):
                 print("No significant error detected in this step.")
            else:
                print("Warning: Unexpected response format from LLM. Continuing evaluation.")
                # Optionally handle unexpected format more robustly

        if not error_found:
            print(f"\nNo decisive errors found by step-by-step analysis in file {json_file}")

        print("\n" + "="*50 + "\n")


# --- Binary Search Method ---

def _construct_binary_search_prompt(problem, answer, chat_segment_content, range_description, upper_half_desc, lower_half_desc):
    """Constructs the prompt for the binary search step."""
    return (
        "You are an AI assistant tasked with analyzing a segment of a multi-agent conversation. Multiple agents are collaborating to address a user query, with the goal of resolving the query through their collective dialogue.\n"
        "Your primary task is to identify the location of the most critical mistake within the provided segment. Determine which half of the segment contains the single step where this crucial error occurs, ultimately leading to the failure in resolving the user’s query.\n"
        f"The problem to address is as follows: {problem}\n"
        f"The Answer for the problem is: {answer}\n" # Included as per original code - remove if ground truth shouldn't be used
        f"Review the following conversation segment {range_description}:\n\n{chat_segment_content}\n\n"
        f"Based on your analysis, predict whether the most critical error is more likely to be located in the upper half ({upper_half_desc}) or the lower half ({lower_half_desc}) of this segment.\n"
        "Please provide your prediction by responding with ONLY 'upper half' or 'lower half'. Remember, your answer should be based on identifying the mistake that directly contributes to the failure in resolving the user's query. If no single clear error is evident, consider the step you believe is most responsible for the failure, allowing for subjective judgment, and base your answer on that."
    )

def _report_binary_search_error(chat_history, step, json_file, is_handcrafted):
    """Reports the identified error step from binary search."""
    index_agent = "role" if is_handcrafted else "name"
    entry = chat_history[step]
    agent_name = entry.get(index_agent, 'Unknown Agent')

    print(f"\nPrediction for {json_file}:")
    print(f"Agent Name: {agent_name}")
    print(f"Step Number: {step}")
    print("\n" + "="*50 + "\n")

def _find_error_in_segment_recursive(client: AzureOpenAI, model: str, max_tokens: int, chat_history: list, problem: str, answer: str, start: int, end: int, json_file: str, is_handcrafted: bool):
    """Recursive helper function for binary search analysis."""
    if start > end:
         print(f"Warning: Invalid range in binary search for {json_file} (start={start}, end={end}). Reporting last valid step.")
         _report_binary_search_error(chat_history, end if end >= 0 else 0, json_file, is_handcrafted) # Report something reasonable
         return
    if start == end:
        _report_binary_search_error(chat_history, start, json_file, is_handcrafted)
        return

    index_agent = "role" if is_handcrafted else "name"

    segment_history = chat_history[start : end + 1]
    if not segment_history:
        print(f"Warning: Empty segment in binary search for {json_file} (start={start}, end={end}). Cannot proceed.")
        _report_binary_search_error(chat_history, start, json_file, is_handcrafted)
        return

    chat_content = "\n".join([
        f"{entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}"
        for entry in segment_history
    ])

    mid = start + (end - start) // 2 

    range_description = f"from step {start} to step {end}"
    upper_half_desc = f"from step {start} to step {mid}"
    lower_half_desc = f"from step {mid + 1} to step {end}"

    prompt = _construct_binary_search_prompt(problem, answer, chat_content, range_description, upper_half_desc, lower_half_desc)

    messages = [
        {"role": "system", "content": "You are an AI assistant specializing in localizing errors in conversation segments."},
        {"role": "user", "content": prompt}
    ]

    print(f"Analyzing step {start}-{end} for {json_file}...")
    result = _make_api_call(client, model, messages, max_tokens)

    if not result:
        print(f"API call failed for segment {start}-{end}. Stopping binary search for {json_file}.")
        return

    print(f"LLM Prediction for segment {start}-{end}: {result}")
    result_lower = result.lower() 

    if "upper half" in result_lower:
         _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, start, mid, json_file, is_handcrafted)
    elif "lower half" in result_lower:
         new_start = min(mid + 1, end)
         _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, new_start, end, json_file, is_handcrafted)
    else:
        print(f"Warning: Ambiguous response '{result}' from LLM for segment {start}-{end}. Randomly choosing a half.")
        if random.randint(0, 1) == 0:
            print("Randomly chose upper half.")
            _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, start, mid, json_file, is_handcrafted)
        else:
            print("Randomly chose lower half.")
            new_start = min(mid + 1, end)
            _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, new_start, end, json_file, is_handcrafted)


def binary_search(client: AzureOpenAI, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int):
    """
    Analyzes chat history using a binary search approach to find the error step.
    """
    print("\n--- Starting Binary Search Analysis ---\n")
    json_files = _get_sorted_json_files(directory_path)

    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        answer = data.get("ground_truth", "") # Keep ground truth if needed

        if not chat_history:
            print(f"Skipping {json_file}: No chat history found.")
            continue

        print(f"--- Analyzing File: {json_file} ---")
        _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, 0, len(chat_history) - 1, json_file, is_handcrafted)
        
        
        
        
# --- Chunk-Parallel Method ---

import re
from typing import List, Tuple, Dict, Any, Optional

CHUNK_SIZE = 20
CHUNK_OVERLAP = 5
FINAL_WINDOW_RADIUS = 3  # ±5 → 총 11 스텝

def _make_chunks(L: int, C: int, O: int) -> List[Tuple[int, int]]:
    """
    Return list of (start, end_exclusive) with overlap.
    Example: L=50, C=20, O=5 -> [(0,20),(15,35),(30,50)]
    """
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
    """
    Render steps [start:end_exclusive) as 'Step i - Agent: content' lines.
    """
    idx_key = "role" if is_handcrafted else "name"
    seg = []
    for i in range(start, end_exclusive):
        entry = chat_history[i]
        agent = entry.get(idx_key, "Unknown Agent")
        content = entry.get("content", "")
        seg.append(f"Step {i} - {agent}: {content}")
    return "\n".join(seg)

def _parse_chunk_response(text: str) -> Tuple[Optional[int], Optional[float], str]:
    """
    Expect formats like:
      Step: 37
      Reason: ...
      Confidence: 0.82
    Return (step, confidence, reason). If parse fails, step/confidence may be None.
    """
    step = None
    conf = None
    reason = ""
    agent= None
    # Step
    m = re.search(r"step\s*[:=]\s*(\d+)", text, flags=re.IGNORECASE)
    if m:
        try:
            step = int(m.group(1))
        except:
            step = None
    # Confidence
    m = re.search(r"confidence\s*[:=]\s*([01](?:\.\d+)?)", text, flags=re.IGNORECASE)
    if m:
        try:
            conf = float(m.group(1))
        except:
            conf = None
    # Reason
    m = re.search(r"reason\s*[:=]\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        reason = m.group(1).strip()
    m = re.search(r"agent\s*[:=]\s*([^\r\n]+)", text, flags=re.IGNORECASE)
    if m: 
        agent = m.group(1).strip()
    m = re.search(r"agent\s*[:=]\s*([^\r\n]+)", text, flags=re.IGNORECASE)
    if m:
        agent = m.group(1).strip()

    return agent, step, conf, reason

def _judge_chunk(
    client: AzureOpenAI,
    model: str,
    max_tokens: int,
    problem: str,
    ground_truth: str,
    segment_text: str,
    chunk_range: Tuple[int, int]
) -> Dict[str, Any]:
    """
    Ask LLM to pick exactly one suspicious step in the chunk, with reason and confidence [0,1].
    Returns dict with keys: {'step', 'confidence', 'reason', 'raw'}
    """
    start, end_ex = chunk_range
    prompt = (
        "You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem.\n"
        f"Problem: {problem}\n"
        f"Answer: {ground_truth}\n"
        f"Consider ONLY the following conversation steps (global indices {start} to {end_ex-1}):\n\n"
        f"{segment_text}\n\n"
        "Select exactly ONE step most likely to be the earliest critical mistake.\n"
        "CALIBRATION RUBRIC for Confidence (strictly follow):\n"
        "- 0.0–0.2 = very weak evidence (speculative, multiple plausible alternatives)\n"
        "- 0.2–0.4 = weak evidence (some hints but contradictions/unknowns remain)\n"
        "- 0.4–0.6 = moderate (clear signals but with notable uncertainty)\n"
        "- 0.6–0.8 = strong (direct indicators, few plausible alternatives)\n"
        "- 0.8–1.0 = very strong (direct contradiction to goal/instructions with explicit traceable impact)\n"
        "If evidence is not explicit and traceable, keep confidence ≤ 0.4.\n\n"
        "Respond ONLY in the format:\n"
        "Agent: <agent name as shown in the segment>\n"
        "Step: <global step integer>\n"
        "Reason: <brief, traceable reason citing concrete step content>\n"
        "Confidence: <number between 0 and 1 with one decimal place>\n"
    )
    messages = [
        {"role": "system", "content": "You are a precise, calibrated LLM judge for error localization."},
        {"role": "user", "content": prompt},
    ]
    resp = _make_api_call(client, model, messages, max_tokens) or ""
    agent, step, conf, reason = _parse_chunk_response(resp)
    # Clamp confidence if provided
    if conf is not None:
        conf = max(0.0, min(1.0, conf))
    # Sanity: ensure the step falls inside the chunk; if not, null it
    if step is not None and not (start <= step < end_ex):
        step = None
    return {"agent": agent, "step": step, "confidence": conf, "reason": reason, "raw": resp, "range": (start, end_ex)}

def _parse_final_response(text: str) -> Tuple[Optional[int], str]:
    """
    Expect:
      Step: <global int>
      Reason: ...
    """
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

def _final_window_judge(
    client: AzureOpenAI,
    model: str,
    max_tokens: int,
    problem: str,
    ground_truth: str,
    chat_history: list,
    is_handcrafted: bool,
    pivot_step: int
) -> Dict[str, Any]:
    """
    Build ±a window around pivot_step and ask LLM to pick the final single step.
    """
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
    resp = _make_api_call(client, model, messages, max_tokens) or ""
    step, reason = _parse_final_response(resp)
    # Constrain to the window
    if step is None or not (win_start <= step <= win_end):
        step = pivot_step  # fallback to the pivot if parsing/window check fails
    return {"step": step, "reason": reason, "raw": resp, "window": (win_start, win_end)}

def chunk_parallel(client: AzureOpenAI, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int):

    print("\n--- Starting Chunk-Parallel Analysis ---\n")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if is_handcrafted else "name"

    for json_file in tqdm(json_files):
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
        # 1) 청크 병렬(실제로는 순차 호출이지만, 로직상 청크 단위 독립)
        per_chunk_results: List[Dict[str, Any]] = []
        for (s, e) in chunks:
            seg_text = _render_segment(chat_history, s, e, is_handcrafted)
            res = _judge_chunk(client, model, max_tokens, problem, ground_truth, seg_text, (s, e))
            per_chunk_results.append(res)
            # 로그 요약
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
        # 2) 신뢰도 기준 Top-1 선택 (conf가 없으면 0 취급)
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

        # 3) 최종 윈도우 judge
        final_res = _final_window_judge(client, model, max_tokens, problem, ground_truth, chat_history, is_handcrafted, pivot)
        final_step = final_res["step"]
        # 최종 에이전트명 추출
        entry = chat_history[final_step]
        agent_name = entry.get(index_agent, "Unknown Agent")

        print(f"\n=== Final Prediction for {json_file} ===")
        print(f"Agent Name: {agent_name}")
        print(f"Step Number: {final_step}")
        print(f"Reason: {final_res.get('reason','')}")
        print(f"Pivot (from chunks): {pivot}, Window: {final_res['window']}")
        print("=" * 50 + "\n")
        
