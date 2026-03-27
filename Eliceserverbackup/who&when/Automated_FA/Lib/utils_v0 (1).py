import os
import json
import random
from openai import AzureOpenAI
from tqdm import tqdm
import math
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import random, threading


##수정 전역 쿨다운 타임스탬프 
_NEXT_ALLOWED_TS = 0.0
_TS_LOCK = threading.Lock()
def _set_cooldown(wait_s: float):
    """지금부터 wait_s초 이후로 전역 쿨다운을 설정"""
    global _NEXT_ALLOWED_TS
    with _TS_LOCK:
        _NEXT_ALLOWED_TS = max(_NEXT_ALLOWED_TS, time.time() + wait_s)

def _get_cooldown_sleep():
    """다음 호출 전 대기해야 할 잔여 시간(초) 계산"""
    with _TS_LOCK:
        remain = _NEXT_ALLOWED_TS - time.time()
    return max(0.0, remain)

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


## 수정
# def _make_api_call(client, model, messages, max_tokens):
#     """Makes an API call to Azure OpenAI."""
#     try:
#         response = client.chat.completions.create(
#             model=model,
#             messages=messages,
#             max_tokens=max_tokens
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         print(f"Error during OpenAI API call: {e}")
#         return None

import time, random, re

# Lib/utils.py: _make_api_call 교체
def _make_api_call(client, model, messages, max_tokens, *, max_retries=6, base_sleep=1.0):
    attempt = 0
    while True:
        try:
            # 호출 전에 전역 쿨다운 존중 (재시도 루프 첫 진입도 포함)
            pre_wait = _get_cooldown_sleep()
            if pre_wait > 0:
                time.sleep(pre_wait)

            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens
            )
            return resp.choices[0].message.content.strip()

        except Exception as e:
            msg = str(e)
            retriable = ("429" in msg) or ("rate limit" in msg.lower()) or any(x in msg for x in ("500","502","503","504"))
            if not retriable or attempt >= max_retries:
                print(f"Error during OpenAI API call (give up): {e}")
                return None

            # 서버가 알려준 대기시간 파싱
            m = re.search(r"try again in\s+([0-9.]+)s", msg.lower())
            wait_s = float(m.group(1)) if m else (base_sleep * (2 ** attempt))
            # 전역 쿨다운에 반영 (+ 약간의 지터)
            wait_s *= (1.0 + 0.2 * random.random())
            wait_s = min(wait_s, 20.0)
            _set_cooldown(wait_s)

            attempt += 1
            print(f"[API RETRY {attempt}] {e} → sleeping {wait_s:.2f}s")
            time.sleep(wait_s)





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
      
      
'''
----------------------------------------------------------------------------------------------------------------------------------------------------------
'''  


# =========================
# === Chunk-Parallel ===
# =========================
import math
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

def _make_chunks(L: int, C: int, O: int):
    """
    Return list of (start, end_exclusive) with overlap.
    end is exclusive to align with Python slicing.
    """
    chunks = []
    if L <= 0:
        return chunks
    s = 0
    while s < L:
        e = min(L, s + C)
        chunks.append((s, e))  # [s:e)
        if e == L:
            break
        s = e - O
        if s < 0:  # safety
            s = 0
            if s == 0 and e == L:
                break
    return chunks

def _render_segment(chat_history: list, start: int, end: int, is_handcrafted: bool):
    """
    Render steps [start:end) as 'Agent: content' lines.
    """
    idx_key = "role" if is_handcrafted else "name"
    seg = []
    for i in range(start, end):
        entry = chat_history[i]
        agent = entry.get(idx_key, "Unknown Agent")
        content = entry.get("content", "")
        seg.append(f"[{i}] {agent}: {content}")
    return "\n".join(seg)

def _build_chunk_prompt(problem: str, chat_segment_text: str, use_synopsis: bool = False,
                        past_synopsis: str | None = None, future_synopsis: str | None = None):
    """
    Prompt for chunk-level judging. Optimized for accurate JSON responses and confidence scoring.
    Does NOT include ground_truth to avoid data leakage.
    """
    syn_txt = ""
    if use_synopsis:
        syn_txt = "\n[Past synopsis]\n" + (past_synopsis or "") + "\n[Future synopsis]\n" + (future_synopsis or "")
    
    prompt = (
        "You are an expert error detection system analyzing multi-agent conversation logs. Your task is to identify the SINGLE most suspicious step within the given chunk that could lead to incorrect problem-solving.\n\n"
        
        "ANALYSIS REQUIREMENTS:\n"
        "1. Examine ONLY the steps within this chunk - do not reference external information\n"
        "2. Identify the most and first suspicious step based on logical errors, incorrect reasoning, or misleading information\n"
        "3. Provide a confidence score (0.0-1.0) based on how certain you are about the error\n"
        "4. Always select exactly one step, even if no clear error exists\n\n"
        
        "CONFIDENCE SCORING GUIDELINES:\n"
        "- 0.8-1.0: Clear logical error, incorrect facts, or obviously wrong reasoning\n"
        "- 0.6-0.8: Suspicious reasoning, potentially misleading information, or questionable approach\n"
        "- 0.4-0.6: Minor issues, suboptimal choices, or unclear explanations\n"
        "- 0.2-0.4: Slight concerns but generally acceptable reasoning\n"
        "- 0.0-0.2: No significant issues identified\n\n"
        
        f"PROBLEM CONTEXT:\n{problem}\n"
        f"{syn_txt}\n"
        f"CONVERSATION CHUNK:\n{chat_segment_text}\n\n"
        
        "OUTPUT REQUIREMENTS:\n"
        "- Respond with ONLY valid JSON in the exact format below\n"
        "- Do not include any explanatory text before or after the JSON\n"
        "- Ensure all fields are present and properly formatted\n\n"
        
        "REQUIRED JSON FORMAT:\n"
        "{\n"
        '  "has_error": true,\n'
        '  "local_step": 3,\n'
        '  "agent": "AgentName",\n'
        '  "confidence": 0.75,\n'
        '  "reason": "Brief explanation of why this step is suspicious"\n'
        "}\n\n"
        
        "Analyze the chunk and respond with JSON only:"
    )
    return prompt

def _safe_float(x, default=1.0):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return max(0.0, min(1.0, v))
    except Exception:
        return default

def _parse_chunk_output(text: str):
    """
    Try to parse strict JSON first. If it fails, attempt enhanced heuristic fallback.
    Returns a normalized dict with unified format.
    """
    if not text:
        return {"has_error": False, "local_step": 0, "agent": "Unknown", "confidence": 0.0, "reason": ""}

    # Clean the text first
    text = text.strip()
    
    # Try JSON parsing first
    try:
        # Try to extract JSON from text if it's wrapped in other content
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)
        else:
            json_text = text
            
        obj = json.loads(json_text)
        if isinstance(obj, dict):
            return {
                "has_error": bool(obj.get("has_error", False)),
                "local_step": max(0, int(obj.get("local_step", 0))),  # Ensure non-negative
                "agent": str(obj.get("agent", "Unknown")).strip(),
                "confidence": _safe_float(obj.get("confidence", 0.0)),
                "reason": str(obj.get("reason", "")).strip(),
            }
    except Exception as e:
        print(f"JSON parsing failed: {e}")

    # Enhanced heuristic fallback
    low = text.lower()
    
    # Parse has_error
    has_err = False
    if "has_error" in low:
        if "true" in low[low.find("has_error"):low.find("has_error")+50]:
            has_err = True
        elif "false" in low[low.find("has_error"):low.find("has_error")+50]:
            has_err = False
    
    # Parse local_step - look for numbers after "local_step" or "step"
    local_step = 0
    step_patterns = [
        r'local_step["\s:]*(\d+)',
        r'step["\s:]*(\d+)',
        r'index["\s:]*(\d+)'
    ]
    for pattern in step_patterns:
        match = re.search(pattern, low)
        if match:
            local_step = max(0, int(match.group(1)))
            break
    
    if local_step == 0:  # Fallback to first number found
        ints = [int(m.group()) for m in re.finditer(r'\d+', text)]
        local_step = ints[0] if ints else 0
    
    # Parse agent name
    agent = "Unknown"
    agent_patterns = [
        r'agent["\s:]*["\']([^"\']+)["\']',
        r'agent["\s:]*([A-Za-z][A-Za-z0-9_]*)',
    ]
    for pattern in agent_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            agent = match.group(1).strip()
            break
    
    # Parse confidence
    confidence = 0.3  # Default low confidence for heuristic parsing
    conf_patterns = [
        r'confidence["\s:]*([0-9]*\.?[0-9]+)',
        r'conf["\s:]*([0-9]*\.?[0-9]+)',
    ]
    for pattern in conf_patterns:
        match = re.search(pattern, low)
        if match:
            confidence = _safe_float(match.group(1), 0.3)
            break
    
    # If has_error is true but confidence is low, boost it slightly
    if has_err and confidence < 0.5:
        confidence = 0.6
    
    # Parse reason
    reason = ""
    reason_patterns = [
        r'reason["\s:]*["\']([^"\']+)["\']',
        r'reason["\s:]*([^,}\n]+)',
        r'explanation["\s:]*["\']([^"\']+)["\']',
    ]
    for pattern in reason_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            reason = match.group(1).strip()
            break
    
    return {
        "has_error": has_err,
        "local_step": local_step,
        "agent": agent,
        "confidence": confidence,
        "reason": reason,
    }

_RATE_SEMAPHORE = None
def _init_rate_semaphore(max_parallel: int):
    global _RATE_SEMAPHORE
    if _RATE_SEMAPHORE is None:
        _RATE_SEMAPHORE = threading.Semaphore(max_parallel)


def _judge_chunk_guarded(client, model, max_tokens, problem, chat_history, s, e, is_handcrafted,
                         use_synopsis=False, stagger_ms=150):
    """
    세마포어로 동시 호출 수를 제한하고, 호출 사이에 약간의 지연(스태거)을 둬 TPM 버스트를 완화.
    """
    global _RATE_SEMAPHORE
    with _RATE_SEMAPHORE:
        # 전역 쿨다운이 남아있으면 먼저 대기
        pre_wait = _get_cooldown_sleep()
        if pre_wait > 0:
            time.sleep(pre_wait)
        # 기본 스태거(소량)
        if stagger_ms > 0:
            time.sleep(stagger_ms / 1000.0)
        return _judge_chunk(client, model, max_tokens, problem, chat_history, s, e, is_handcrafted, use_synopsis)

def _judge_chunk(client, model: str, max_tokens: int, problem: str,
                 chat_history: list, start: int, end: int, is_handcrafted: bool,
                 use_synopsis: bool = False):
    """
    Call LLM once for the chunk [start:end) and parse JSON result.
    """
    seg_txt = _render_segment(chat_history, start, end, is_handcrafted)
    # (요약은 초기엔 OFF 권장)
    prompt = _build_chunk_prompt(problem, seg_txt, use_synopsis=use_synopsis)
    messages = [
        {"role": "system", "content": "You are a careful and concise error localizer."},
        {"role": "user", "content": prompt}
    ]
    resp = _make_api_call(client, model, messages, max_tokens)
    parsed = _parse_chunk_output(resp)
    return parsed

# 기존 _run_chunks_parallel 교체
def _run_chunks_parallel(client, model, max_tokens, problem, chat_history, chunks, is_handcrafted,
                         use_synopsis=False, max_workers=1, stagger_ms=150):
    """
    Run chunk judges with concurrency throttling and small staggering to avoid 429.
    """
    _init_rate_semaphore(max_workers)
    outputs = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut2idx = {}
        for idx, (s, e) in enumerate(chunks):
            fut = ex.submit(
                _judge_chunk_guarded,
                client, model, max_tokens, problem,
                chat_history, s, e, is_handcrafted,
                use_synopsis, stagger_ms
            )
            fut2idx[fut] = idx
        for fut in as_completed(fut2idx):
            idx = fut2idx[fut]
            try:
                outputs[idx] = fut.result()
            except Exception as e:
                print(f"[Chunk {idx}] judge error: {e}")
                outputs[idx] = {"has_error": False, "local_step": 0, "agent": "Unknown", "confidence": 0.0, "reason": ""}
    return outputs


def _chunk_center_weight(local_step: int, chunk_len: int, beta: float = 0.2):
    """
    1 - beta * normalized distance from chunk center. In [0.0, 1.0].
    """
    if chunk_len <= 1:
        return 1.0
    # local_step이 범위를 벗어날 수 있어 LLM 출력 등을 대비해 클램프
    ls = max(0, min(local_step, chunk_len - 1))
    center = (chunk_len - 1) / 2.0
    denom = max(1.0, center, 1e-8)      # 분모 epsilon 가드
    dist = abs(ls - center) / denom
    # beta가 실수로 크게 설정되어도 음수로 내려가지 않도록 하한 0.0
    return max(0.0, 1.0 - beta * dist)

def _merge_candidates(chunks, outputs):
    """
    Merge chunk results into candidates based on confidence.
    All chunks now provide confidence-based results regardless of has_error.
    """
    candidates = {}  # global_step -> {"conf_list":[], "agents":[], "reasons":[]}

    for (s, e), out in zip(chunks, outputs):
        if not out:
            continue
        
        local_step = int(out.get("local_step", 0))
        global_step = s + local_step
        conf = _safe_float(out.get("confidence", 0.0))
        agent = str(out.get("agent", "Unknown"))
        reason = str(out.get("reason", ""))
        
        bucket = candidates.setdefault(global_step, {"conf_list": [], "agents": [], "reasons": []})
        bucket["conf_list"].append(conf)
        bucket["agents"].append(agent)
        bucket["reasons"].append(reason)

    # normalize to lists
    cand_list = []
    for g, v in candidates.items():
        cand_list.append({
            "global_step": g,
            "avg_conf": sum(v["conf_list"]) / max(1, len(v["conf_list"])),
            "max_conf": max(v["conf_list"]) if v["conf_list"] else 0.0,
            "agents": v["agents"],
            "reasons": v["reasons"],
        })

    return cand_list

def _pick_best_from_candidates(cand_list, L: int):
    """
    Pick best candidate by avg_conf; tie-breakers:
      1) closest to the middle of document
    """
    if not cand_list:
        return None
    # primary: avg_conf desc
    cand_list = sorted(cand_list, key=lambda x: (-x["avg_conf"], abs(x["global_step"] - (L-1)/2.0)))
    best = cand_list[0]
    # choose most frequent agent (majority)
    agents = best.get("agents", [])
    reasons = best.get("reasons", [])
    agent = "Unknown"
    reason = ""
    if agents:
        counts = {}
        for a in agents:
            counts[a] = counts.get(a, 0) + 1
        agent = max(counts.items(), key=lambda kv: kv[1])[0]
    if reasons:
        # Pick the first non-empty reason
        reason = next((r for r in reasons if r.strip()), "")
    return {"global_step": best["global_step"], "agent": agent, "score": best["avg_conf"], "reason": reason}


def _build_refine_prompt(problem: str, window_text: str):
    return (
        "You are a precise judge. Given a SHORT window of the multi-agent log, pick the SINGLE step (inside the window) "
        "that is the earliest decisive mistake that leads to the wrong final answer. Output STRICT JSON:\n"
        '{ "local_step": <int>, "agent": "<string>", "reason": "<short explanation>" }\n'
        "Do not add extra keys.\n"
        f"[Problem]\n{problem}\n"
        "[Window]\n"
        f"{window_text}\n"
        "Now output ONLY the JSON."
    )

def _final_refine(client, model: str, max_tokens: int, problem: str, chat_history: list,
                  center_step: int, delta: int, is_handcrafted: bool):
    """
    One extra judge on [center-delta : center+delta] inclusive window. Return (global_step, agent).
    """
    L = len(chat_history)
    s = max(0, center_step - delta)
    e = min(L - 1, center_step + delta)
    # render inclusive [s:e] -> use exclusive upper for rendering
    window_txt = _render_segment(chat_history, s, e + 1, is_handcrafted)
    prompt = _build_refine_prompt(problem, window_txt)
    messages = [
        {"role": "system", "content": "You are a careful and concise error localizer."},
        {"role": "user", "content": prompt}
    ]
    resp = _make_api_call(client, model, messages, max_tokens)
    # parse very small JSON
    try:
        obj = json.loads(resp)
        local_step = int(obj.get("local_step", 0))
        agent = str(obj.get("agent", "Unknown"))
        reason = str(obj.get("reason", ""))
    except Exception:
        # fallback: pick center
        local_step = center_step - s
        agent = "Unknown"
        reason = ""
    global_step = s + local_step
    return global_step, agent, reason

def chunk_parallel(client, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int,
                   chunk_size: int = 20, overlap: int = 5, delta_refine: int = 5,
                   use_synopsis: bool = False, max_workers: int = 1):
    """
    Entry point for Hand-crafted dataset: Chunked parallel judging + merge + final refine.
    """
    print("\n--- Starting Chunk-Parallel Analysis ---\n")
    json_files = _get_sorted_json_files(directory_path)
    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")

        if not chat_history:
            print(f"Skipping {json_file}: No chat history found.")
            continue

        L = len(chat_history)
        # Only run this pipeline for hand-crafted (길이 긴 로그)
        if not is_handcrafted:
            print(f"Skipping {json_file}: chunk_parallel is intended for Hand-crafted logs.")
            continue

        print(f"--- Analyzing File: {json_file} ---")
        chunks = _make_chunks(L, chunk_size, overlap)
        if not chunks:
            print(f"No chunks produced for {json_file}.")
            continue

        # 1) 병렬 판정
        outs = _run_chunks_parallel(client, model, max_tokens, problem, chat_history, chunks,
                                    is_handcrafted=is_handcrafted, use_synopsis=use_synopsis, max_workers=max_workers)

        # 2) 통합
        cand_list = _merge_candidates(chunks, outs)

        # 3) 최종 후보 선택 (confidence 기반)
        best = _pick_best_from_candidates(cand_list, L)

        # 4) 후보가 없는 경우 센터 추정
        if best is None:
            best = {"global_step": L // 2, "agent": "Unknown", "score": 0.0, "reason": ""}

        # 5) 재확인 1회 (±delta_refine)
        final_step, final_agent, final_reason = _final_refine(
            client, model, max_tokens, problem, chat_history, best["global_step"], delta_refine, is_handcrafted
        )

        # 6) 리포트
        idx_key = "role" if is_handcrafted else "name"
        entry = chat_history[final_step]
        inferred_agent = entry.get(idx_key, final_agent or "Unknown")

        print(f"\nPrediction for {json_file}:")
        print(f"Agent Name: {inferred_agent}")
        print(f"Step Number: {final_step}")
        
        # reason 출력 - best에서 가져온 reason과 final_reason 모두 확인
        reason_to_show = final_reason or best.get("reason", "")
        if reason_to_show:
            print(f"Reason: {reason_to_show}")
        print("\n" + "="*50 + "\n")
