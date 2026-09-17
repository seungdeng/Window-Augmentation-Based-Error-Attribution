import os
import json
from openai import OpenAI
from tqdm import tqdm
# --- Helper Functions ---
# Lib/utils.py

import re
from typing import Tuple, Dict, Any, Optional

def _parse_stage1_response(text: str):
    """
    Parse the 1-stage (global) prediction text produced by all_at_once():
      Agent Name: <name>
      Step Number: <int>
      Reason for Mistake: <free text>
    Returns: (agent:str|None, step:int|None, reason:str)
    Robust to minor wording variations (e.g., 'Reason:' vs 'Reason for Mistake:').
    """
    if not isinstance(text, str) or not text.strip():
        return (None, None, "")

    agent = None
    step = None
    reason = ""

    # Agent line
    m = re.search(r"Agent\s*Name\s*:\s*([^\r\n]+)", text, flags=re.IGNORECASE)
    if m:
        agent = m.group(1).strip()

    # Step line (integer)
    m = re.search(r"Step\s*Number\s*:\s*(\d+)", text, flags=re.IGNORECASE)
    if m:
        try:
            step = int(m.group(1))
        except Exception:
            step = None

    # Reason line (accept both "Reason:" and "Reason for Mistake:")
    m = re.search(r"(Reason(?:\s*for\s*Mistake)?)\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        reason = m.group(2).strip()

    return (agent, step, reason)

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
    """Makes a chat-completion call via an OpenAI-compatible API (e.g. OpenRouter)."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.6,
            seed=42,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during LLM API call: {e}")
        return None

# --- All-at-Once Method ---

def all_at_once(client: OpenAI, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int):
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

def step_by_step(client: OpenAI, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int):
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


# --- Window-Based Final Judge (shared helper for the window-augmentation methods) ---

FINAL_WINDOW_RADIUS = 3  # ±5 → 총 11 스텝

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

def _parse_final_response(text: str) -> Tuple[Optional[int], str]:
    """
    Expect:
      Step: <global int>
      Reason: ...
    """
    agent=None
    step = None
    reason = ""

    # Agent
    m = re.search(r"agent\s*[:=]\s*([^\r\n]+)", text, flags=re.IGNORECASE)
    if m:
        agent = m.group(1).strip()

    m = re.search(r"step\s*[:=]\s*(\d+)", text, flags=re.IGNORECASE)
    if m:
        try:
            step = int(m.group(1))
        except:
            step = None
    m = re.search(r"reason\s*[:=]\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        reason = m.group(1).strip()
    return agent, step, reason

def _final_window_judge(
    client: OpenAI,
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
        "Agent: <exact agent name>\n"
        "Step: <global step integer>\n"
        "Reason: <short reason>\n"
    )
    messages = [
        {"role": "system", "content": "You are a decisive final judge for pinpointing a single critical mistake step."},
        {"role": "user", "content": prompt},
    ]
    resp = _make_api_call(client, model, messages, max_tokens) or ""
    agent, step, reason = _parse_final_response(resp)
    # Constrain to the window
    if step is None or not (win_start <= step <= win_end):
        step = pivot_step  # fallback to the pivot if parsing/window check fails
    return {"agent": agent, "step": step, "reason": reason, "raw": resp, "window": (win_start, win_end)}

def all_at_once_with_window(client: OpenAI, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int):
    """
    1) 전체 대화로 1-stage 예측(Agent/Step/Reason)
    2) 예측 Step을 pivot으로 ±FINAL_WINDOW_RADIUS 윈도우를 만들어 최종 판정
    """
    print("\n--- Starting All-at-Once(+Window) Analysis ---\n")
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

        chat_content = "\n".join([
            f"{entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}"
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

        # 1-stage 호출
        stage1 = _make_api_call(client, model, messages, max_tokens) or ""
        print(f"Prediction for {json_file}:")
        print(stage1 if stage1 else "Failed to get prediction.")
        print("\n" + "="*50 + "\n")

        # 파싱 → pivot
        agent_1, step_1, reason_1 = _parse_stage1_response(stage1)
        L = len(chat_history)
        pivot = step_1 if isinstance(step_1, int) and 0 <= step_1 < L else (L // 2)

        # 2-stage 최종 판정 (윈도우 기반 최종 판정 재사용)
        final_res = _final_window_judge(
            client=client,
            model=model,
            max_tokens=max_tokens,
            problem=problem,
            ground_truth=ground_truth,
            chat_history=chat_history,
            is_handcrafted=is_handcrafted,
            pivot_step=pivot
        )
        final_step = final_res["step"]
        # 최종 agent
        final_agent = final_res.get("agent")  # ← LLM이 최종 선택한 에이전트 그대로

        print(f"=== Final Prediction for {json_file} ===")
        print(f"Agent Name: {final_agent}")
        print(f"Step Number: {final_step if final_step is not None else pivot}")
        print(f"Reason: {final_res.get('reason','').strip() or reason_1}")
        print(f"Pivot (from stage1): {pivot}, Window: {final_res['window']}")
        print("=" * 50 + "\n")


def step_by_step_with_window(client: OpenAI, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int):
    """
    1) 스텝마다 Yes/No 평가해 최초 'Yes' 스텝을 pivot으로 확정
    2) pivot 주변 윈도우로 최종 판정
    """
    print("\n--- Starting Step-by-Step(+Window) Analysis ---\n")
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
        current_conversation_history = ""
        found = False
        pivot = None
        first_agent = None
        first_reason = ""

        print(f"--- Analyzing File: {json_file} ---")
        for idx, entry in enumerate(chat_history):
            agent_name = entry.get(index_agent, 'Unknown Agent')
            content = entry.get('content', '')
            current_conversation_history += f"Step {idx} - {agent_name}: {content}\n"

            prompt = (
                f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. "
                f"The problem being addressed is: {problem}. "
                f"The Answer for the problem is: {ground_truth}\n"
                f"Here is the conversation history up to the current step:\n{current_conversation_history}\n"
                f"The most recent step ({idx}) was by '{agent_name}'.\n"
                f"Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution. "
                "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
                "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process. "
                "Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
            )

            messages = [
                {"role": "system", "content": "You are a precise step-by-step conversation evaluator."},
                {"role": "user", "content": prompt},
            ]

            print(f"Evaluating Step {idx} by {agent_name}...")
            answer = _make_api_call(client, model, messages, max_tokens)

            if not answer:
                print("Failed to get evaluation for this step. Stopping analysis for this file.")
                found = True
                break

            print(f"LLM Evaluation: {answer}")
            lower = answer.strip().lower()

            if lower.startswith("1. yes"):
                # 1-stage 발견
                try:
                    first_reason = answer.split("Reason:", 1)[-1].strip()
                except Exception:
                    first_reason = "[Could not extract reason]"
                pivot = max(0, min(idx, L - 1))
                first_agent = agent_name

                print(f"\nPrediction for {json_file}: Error found.")
                print(f"Agent Name: {first_agent}")
                print(f"Step Number: {pivot}")
                print(f"Reason provided by LLM: {first_reason}")

                # 2-stage 최종 판단
                final_res = _final_window_judge(
                    client=client,
                    model=model,
                    max_tokens=max_tokens,
                    problem=problem,
                    ground_truth=ground_truth,
                    chat_history=chat_history,
                    is_handcrafted=is_handcrafted,
                    pivot_step=pivot
                )
                final_step = final_res["step"]
                final_agent = final_res.get("agent")  # ← LLM이 최종 선택한 에이전트 그대로

                print(f"\n=== Final Prediction for {json_file} ===")
                print(f"Agent Name: {final_agent}")
                print(f"Step Number: {final_step if final_step is not None else pivot}")
                print(f"Reason: {final_res.get('reason','').strip() or first_reason}")
                print(f"Pivot (from step-by-step): {pivot}, Window: {final_res['window']}")
                print("=" * 50 + "\n")

                found = True
                break

            elif lower.startswith("1. no"):
                continue
            else:
                print(f"Warning: Unexpected response format at step {idx}: {answer[:100]}...")

        if not found:
            print(f"\nNo decisive errors found by step-by-step analysis in file {json_file}")
            print("\n" + "="*50 + "\n")
