"""
Trajectory-Repair experiment (paper extension, not the patent-grade design).

Given the decisive error step/agent already identified by the existing
window-augmentation attribution methodology (Process C/D in main_exp), this
module:
  1) (optional, off by default) explains what was procedurally wrong with
     that step -- see "Two reason modes" below,
  2) asks an LLM playing that agent to produce a corrected version of the step,
  3) regenerates every step after it (same agent order as the original failed
     trajectory, but freshly generated content) WITHOUT ever showing the model
     the original post-error content or the ground truth,
  4) verifies whether the regenerated trajectory's final output now matches
     the ground truth (this is the only other place ground truth is used --
     scoring, not generation).

Two reason modes for step (1) -- controlled by whether the caller passes a
`ground_truth` into the reason-generation step at all (see repair.py's
`--use_gt_reason` flag, OFF by default):

  * DEFAULT, GT-free (`sanitized_reason=None` passed to repair_step): no
    ground truth ever enters the pipeline, anywhere, full stop. The repair
    step is only told "this step was flagged as the decisive error" and must
    self-diagnose the flaw from the task and its own prior output. Simpler,
    cheaper (skips the whole reason-generation call), and leak-proof by
    construction rather than by filtering. This is the recommended default
    and the one closest to a real deployment (no oracle answer available).

  * OPT-IN, GT-informed (`--use_gt_reason`): explain_failure_without_leaking_answer()
    uses ground_truth to produce a more specific diagnosis, but that text is
    the ONLY channel through which anything GT-derived could reach the repair
    phase, so it passes through a hard leak gate (literal check + LLM judge +
    bounded retries + a GT-free template fallback) before being used. Useful
    as a secondary ablation ("how much does an explicit diagnosis help, on
    top of just knowing the location?"), not as the headline number.

Ground-truth usage map (important for the experiment to be valid):
  - Steps 0..s*-1 (context)      : reused verbatim from the original log. No GT.
  - Reason-for-failure generation : GT-free by default (see above); GT used
                                     only under --use_gt_reason, and gated.
  - Repair of step s*             : NO GT, ever. Only prior context + the
                                     agent's own flawed output (+ optionally
                                     the gated reason string).
  - Regeneration of steps > s*    : NO GT, and NOT given the original content
                                     of those steps (that would leak the
                                     "solution" the trajectory is supposed to
                                     rediscover on its own).
  - Verifier                      : GT IS used, but only to grade the final
                                     output after generation is complete.
"""

import os
import re
import json
from typing import Dict, List, Optional, Tuple, Any

# Reuse existing helpers instead of duplicating them.
from Lib.utils import _make_api_call, _get_sorted_json_files, _load_json_data  # noqa: F401


# -------------------------------------------------------------------------
# Reading the decisive-error pivot (s*, agent) for each file
# -------------------------------------------------------------------------

def read_pivot_predictions(eval_file: str) -> Dict[str, Dict[str, str]]:
    """
    Parse a Process C/D attribution log (produced by main_exp's
    `inference.py --two_stage_window`) for blocks like:

      === Final Prediction for 1.json ===
      Agent Name: WebSurfer
      Step Number: 4
      Reason: ...

    Returns: {"<name>.json": {"predicted_agent": str, "predicted_step": str}}
    """
    if not os.path.exists(eval_file):
        print(f"Error: Attribution log not found at {eval_file}")
        return {}

    with open(eval_file, "r", encoding="utf-8") as f:
        data = f.read().replace("\r\n", "\n")

    block_pat = re.compile(
        r"(?:^|\n)===\s*Final Prediction for\s+([^\n:]+?\.json)\s*===\s*"
        r"(.*?)"
        r"(?=(?:^|\n)===\s*Final Prediction for\s+[^\n:]+?\.json\s*===|\Z)",
        re.DOTALL | re.IGNORECASE
    )
    agent_pat = re.compile(r"Agent\s*Name\s*:\s*([^\r\n]+)", re.IGNORECASE)
    step_pat = re.compile(r"Step\s*Number\s*:\s*(\d+)", re.IGNORECASE)

    predictions: Dict[str, Dict[str, str]] = {}
    for m in block_pat.finditer(data):
        fname = m.group(1).strip()
        body = m.group(2)
        a = agent_pat.search(body)
        s = step_pat.search(body)
        if not s:
            continue
        predictions[fname] = {
            "predicted_agent": (a.group(1).strip() if a else ""),
            "predicted_step": s.group(1).strip(),
        }

    print(f"--- Pivot predictions read from {eval_file}: {len(predictions)} files ---")
    return predictions


def get_oracle_pivot(labeled_json: Dict[str, Any]) -> Optional[Tuple[int, str]]:
    """Ground-truth (mistake_step, mistake_agent) pivot -- used only for the oracle upper-bound run."""
    step_raw = labeled_json.get("mistake_step")
    agent = labeled_json.get("mistake_agent")
    try:
        step = int(str(step_raw).strip())
    except (TypeError, ValueError):
        return None
    if agent is None:
        return None
    return step, str(agent)


def attribution_is_correct(s_star: int, mistake_step_raw: Any) -> Optional[bool]:
    """
    Was the pivot step used for repair (s_star) the same step the dataset's own
    label (mistake_step) says is the decisive error? None if mistake_step is
    missing/unparsable (e.g. a corrupted label) -- callers should treat that as
    "unknown", not as a failure.

    Used for the ablation: repair rate when attribution was exactly right vs.
    when it was wrong.
    """
    try:
        return int(s_star) == int(str(mistake_step_raw).strip())
    except (TypeError, ValueError):
        return None


def get_random_pivot(labeled_json: Dict[str, Any], seed_key: str, index_agent: str) -> Optional[Tuple[int, str]]:
    """Deterministic pseudo-random pivot (seeded by filename) -- control baseline."""
    import hashlib
    hist = labeled_json.get("history", [])
    if not hist:
        return None
    h = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest(), 16)
    idx = h % len(hist)
    agent = (hist[idx] or {}).get(index_agent, "Unknown Agent")
    return idx, agent


# -------------------------------------------------------------------------
# Rendering helpers
# -------------------------------------------------------------------------

def _render_range(chat_history: list, start: int, end_exclusive: int, index_agent: str) -> str:
    lines = []
    for i in range(max(0, start), min(end_exclusive, len(chat_history))):
        entry = chat_history[i]
        agent = entry.get(index_agent, "Unknown Agent")
        content = entry.get("content", "")
        lines.append(f"Step {i} - {agent}: {content}")
    return "\n".join(lines) if lines else "(no prior steps)"


# -------------------------------------------------------------------------
# Leakage gate -- this is the ONLY channel through which anything derived
# from ground_truth can reach the repair/regeneration phases, so it is
# checked, not just prompted-away. Anything that fails BOTH the literal
# check and the LLM-judge check is discarded and replaced by a fixed,
# template-only fallback that is leak-free by construction (it contains no
# text derived from ground_truth or from this call's own output).
# -------------------------------------------------------------------------

_MAX_REASON_ATTEMPTS = 3


def _contains_ground_truth(text: str, ground_truth: str) -> bool:
    """Literal/near-literal leak check: does `text` contain the ground-truth
    answer as a standalone token or phrase (case-insensitive)?"""
    gt = str(ground_truth or "").strip()
    if not gt or not text:
        return False
    try:
        return re.search(r"(?<!\w)" + re.escape(gt) + r"(?!\w)", text, re.IGNORECASE) is not None
    except re.error:
        return gt.lower() in text.lower()


def _judge_leaks_answer(client, model: str, max_tokens: int, ground_truth: str, text: str) -> bool:
    """LLM-judge leak check: catches paraphrases/hints the literal check misses."""
    prompt = (
        f"Reference answer (secret -- must not appear below): {ground_truth}\n\n"
        f"Text to check:\n{text}\n\n"
        "Does the text above state, restate, paraphrase, numerically/textually imply, "
        "or otherwise make it possible to directly infer the reference answer? "
        "Answer with exactly one word: YES or NO. If in doubt, answer YES."
    )
    messages = [
        {"role": "system", "content": "You are a strict leakage detector for a fairness-critical experiment. Be conservative."},
        {"role": "user", "content": prompt},
    ]
    resp = _make_api_call(client, model, messages, max_tokens=8) or ""
    return resp.strip().upper().startswith("Y")


def _safe_fallback_reason(agent_name: str) -> str:
    """Guaranteed leak-free reason: contains nothing derived from ground_truth
    or from any model output, used only if every generation attempt leaks."""
    return (
        f"The step attributed to {agent_name} contains a procedural or logical "
        "error that contributed to the task's failure. Re-examine this step's "
        "own reasoning, tool usage, and any data it relied on for mistakes or "
        "unjustified assumptions, and produce a corrected version -- without "
        "assuming or targeting any specific final value."
    )


# -------------------------------------------------------------------------
# Phase A: sanitized failure explanation (GT used, but not leaked forward)
# -------------------------------------------------------------------------

def explain_failure_without_leaking_answer(
    client, model: str, max_tokens: int,
    query: str, ground_truth: str,
    chat_history: list, s_star: int, index_agent: str,
) -> str:
    agent_name = chat_history[s_star].get(index_agent, "Unknown Agent")
    original_content = chat_history[s_star].get("content", "")
    full_trajectory = _render_range(chat_history, 0, len(chat_history), index_agent)

    base_prompt = (
        "You are analyzing why a multi-agent system failed to solve a task.\n"
        f"Task: {query}\n"
        f"Correct final answer (for your reference only): {ground_truth}\n\n"
        f"Full failed trajectory:\n{full_trajectory}\n\n"
        f"The following step has been identified as the earliest decisive error "
        f"(fixing it is expected to give the team the best chance of reaching the "
        f"correct answer):\n"
        f"Step {s_star} - {agent_name}: {original_content}\n\n"
        "In 2-4 sentences, explain what is procedurally or logically wrong with "
        "this step's output (e.g. wrong tool usage, misread constraint, faulty "
        "reasoning, incorrect data extraction).\n"
        "IMPORTANT: Do NOT state, restate, hint at, or numerically/textually imply "
        "the correct final answer anywhere in your explanation. Describe only the "
        "flaw in the process, not the solution."
    )

    for attempt in range(_MAX_REASON_ATTEMPTS):
        prompt = base_prompt
        if attempt > 0:
            prompt += (
                "\n\nYour previous attempt violated this rule by revealing (or "
                "making it possible to infer) the correct answer. Try again: "
                "describe ONLY the procedural flaw, using none of the reference "
                "answer's wording or value, not even partially."
            )
        messages = [
            {"role": "system", "content": "You explain process errors without ever revealing task answers."},
            {"role": "user", "content": prompt},
        ]
        reason = (_make_api_call(client, model, messages, max_tokens) or "").strip()
        if not reason:
            continue
        if _contains_ground_truth(reason, ground_truth):
            continue
        if _judge_leaks_answer(client, model, max_tokens, ground_truth, reason):
            continue
        return reason

    # Every attempt leaked or failed -- fall back to a template that is
    # leak-free by construction rather than passing anything risky downstream.
    return _safe_fallback_reason(agent_name)


# -------------------------------------------------------------------------
# Phase B: repair the decisive-error step itself (NO ground truth)
# -------------------------------------------------------------------------

def repair_step(
    client, model: str, max_tokens: int,
    query: str, chat_history: list, s_star: int, index_agent: str,
    sanitized_reason: Optional[str] = None,
) -> str:
    """
    Correct the decisive-error step. `sanitized_reason` is optional:
      - None (default, GT-free pipeline): the model is only told this step was
        flagged as the likely decisive error and must diagnose the problem
        itself from the task + its own prior output. No information derived
        from ground_truth ever reaches this function.
      - a string (only when the caller opted into --use_gt_reason): a
        leak-gated explanation of the flaw, produced by
        explain_failure_without_leaking_answer().
    """
    agent_name = chat_history[s_star].get(index_agent, "Unknown Agent")
    original_content = chat_history[s_star].get("content", "")
    prior_context = _render_range(chat_history, 0, s_star, index_agent)

    if sanitized_reason:
        diagnosis_block = f"Why it was flawed: {sanitized_reason}\n\n"
    else:
        diagnosis_block = (
            "This step has been flagged as the likely earliest decisive error in "
            "this trajectory -- i.e. the team's failure to complete the task is "
            "believed to trace back to something wrong with this specific step. "
            "No further diagnosis is given: re-examine the step yourself against "
            "the task and the conversation so far, and identify what is likely "
            "wrong (e.g. a misread constraint, a logic or tool-usage mistake, an "
            "unjustified assumption) before correcting it.\n\n"
        )

    prompt = (
        f"Task: {query}\n\n"
        f"Conversation so far (steps 0 to {s_star - 1}):\n{prior_context}\n\n"
        f"You ({agent_name}) are about to (re-)take step {s_star}. Your previous "
        f"attempt at this step was flawed:\n"
        f"<<<PREVIOUS OUTPUT>>>\n{original_content}\n<<<END PREVIOUS OUTPUT>>>\n\n"
        f"{diagnosis_block}"
        "Write a corrected version of your output for this same step. Keep the "
        "same role, tone and output style as your previous attempt (e.g. same "
        "use of code blocks, same first-person planning style) but fix what you "
        "identify as wrong, and anything strictly necessary for that fix to "
        "work. Do not restructure unrelated parts. Do not reference this "
        "instruction, the failure, or the correction process -- write it as a "
        "natural first attempt at this step. Do not perform or narrate steps "
        "after this one.\n\n"
        "Output the corrected step content only."
    )
    messages = [
        {"role": "system", "content": f"You are {agent_name}, an agent in a multi-agent system. Adopt this identity exactly, based only on how it behaved earlier in this conversation."},
        {"role": "user", "content": prompt},
    ]
    corrected = _make_api_call(client, model, messages, max_tokens) or original_content
    return corrected.strip()


# -------------------------------------------------------------------------
# Phase C: regenerate everything after s* (NO ground truth, NO original content)
# -------------------------------------------------------------------------

def regenerate_downstream(
    client, model: str, max_tokens: int,
    query: str, chat_history: list, s_star: int, index_agent: str,
    corrected_content: str,
) -> List[Dict[str, str]]:
    """
    Reuses the ORIGINAL agent order for steps > s* (structural fact, not the
    "answer"), but regenerates each agent's content from scratch -- the
    original content of those steps is never shown to the model.

    Returns a list of {"name"/"role": agent, "content": text} entries for
    steps s*..end (index 0 of the list == corrected s*).
    """
    regenerated = [{index_agent: chat_history[s_star].get(index_agent, "Unknown Agent"), "content": corrected_content}]

    for i in range(s_star + 1, len(chat_history)):
        agent_name = chat_history[i].get(index_agent, "Unknown Agent")
        context_lines = [f"Step {j} - {chat_history[j].get(index_agent, 'Unknown Agent')}: {chat_history[j].get('content', '')}"
                          for j in range(0, s_star)]
        context_lines += [f"Step {s_star + k} - {e.get(index_agent, 'Unknown Agent')}: {e.get('content', '')}"
                           for k, e in enumerate(regenerated)]
        context = "\n".join(context_lines) if context_lines else "(no prior steps)"

        prompt = (
            f"Task: {query}\n\n"
            f"Conversation so far:\n{context}\n\n"
            f"Continue the conversation as {agent_name} (step {i}). Stay "
            f"consistent with this agent's role and the conversation so far. "
            "If, from this point, you believe the task is now solved, clearly "
            "state the final answer. If a prior step in this conversation ended "
            "the task (e.g. a TERMINATE-style message), you may simply confirm "
            "completion."
        )
        messages = [
            {"role": "system", "content": f"You are {agent_name}, continuing a multi-agent conversation to solve a task."},
            {"role": "user", "content": prompt},
        ]
        content = _make_api_call(client, model, messages, max_tokens) or ""
        regenerated.append({index_agent: agent_name, "content": content.strip()})

    return regenerated


# -------------------------------------------------------------------------
# Phase D: verifier (GT used for scoring only)
# -------------------------------------------------------------------------

def verify_success(
    client, model: str, max_tokens: int,
    query: str, ground_truth: str, final_output: str,
) -> Tuple[bool, str]:
    prompt = (
        f"Task: {query}\n"
        f"Reference answer: {ground_truth}\n"
        f"Agent's final answer/output: {final_output}\n\n"
        "Does the agent's final answer match the reference answer (allowing for "
        "reasonable paraphrase, formatting, or unit differences, but not a "
        "substantively different value)?\n"
        "Respond with exactly one line starting with 'Verdict: Success' or "
        "'Verdict: Fail', followed by a one-sentence justification."
    )
    messages = [
        {"role": "system", "content": "You are a strict grader comparing a candidate answer to a reference answer."},
        {"role": "user", "content": prompt},
    ]
    resp = _make_api_call(client, model, messages, max_tokens) or ""
    success = bool(re.search(r"verdict\s*:\s*success", resp, re.IGNORECASE))
    return success, resp.strip()
