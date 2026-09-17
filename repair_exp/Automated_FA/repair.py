"""
Trajectory-repair experiment entry point.

Takes the decisive error step (s*) already identified for each file -- either
from an existing Process C/D attribution log, from the dataset's own
mistake_step/mistake_agent label (oracle upper bound), or from a deterministic
random step (control baseline) -- repairs that step, regenerates everything
after it, and verifies whether the task now succeeds. Ground-truth handling
follows the rules documented at the top of Lib/repair_utils.py.

Usage (from main_exp/../repair_exp/Automated_FA):

    python repair.py --pivot_source attribution_log \
        --attribution_log "outputs/reference/all_at_once_gpt-4o_alg_generated (WIN3).txt" \
        --directory_path ../Who_and_When/Algorithm-Generated \
        --model openai/gpt-4o --limit 5

    python repair.py --pivot_source oracle --model openai/gpt-4o
    python repair.py --pivot_source random --model openai/gpt-4o
"""

import os
import sys
import json
import argparse
import contextlib
import datetime
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI

from Lib.utils import _get_sorted_json_files, _load_json_data
from Lib.repair_utils import (
    read_pivot_predictions,
    get_oracle_pivot,
    get_random_pivot,
    attribution_is_correct,
    explain_failure_without_leaking_answer,
    repair_step,
    regenerate_downstream,
    verify_success,
)

OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def build_client(args):
    cli_key = (args.api_key or "").strip()
    api_key = cli_key or (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        print("Error: Provide --api_key or set OPENROUTER_API_KEY in your .env file.")
        sys.exit(1)

    base_url = (
        (args.base_url or "").strip()
        or (os.getenv("OPENROUTER_BASE_URL") or "").strip()
        or OPENROUTER_DEFAULT_BASE_URL
    )

    default_headers = {}
    site_url = (os.getenv("OPENROUTER_SITE_URL") or "").strip()
    site_name = (os.getenv("OPENROUTER_SITE_NAME") or "").strip()
    if site_url:
        default_headers["HTTP-Referer"] = site_url
    if site_name:
        default_headers["X-Title"] = site_name

    return OpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers or None), base_url


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Repair a failed trajectory from its decisive error step onward, and measure recovery rate."
    )
    parser.add_argument(
        "--pivot_source", choices=["attribution_log", "oracle", "random"], default="attribution_log",
        help="Where the decisive-error step/agent (s*) comes from: the proposed method's "
             "attribution log (main experiment), the dataset's own label (oracle upper "
             "bound), or a deterministic random step (control baseline)."
    )
    parser.add_argument(
        "--attribution_log", type=str, default="",
        help="Path to a Process C/D log, e.g. 'outputs/reference/all_at_once_gpt-4o_alg_generated (WIN3).txt'. "
             "Required when --pivot_source=attribution_log."
    )
    parser.add_argument("--directory_path", type=str, default="../Who_and_When/Algorithm-Generated",
                         help="Path to the directory containing JSON chat history files.")
    parser.add_argument("--is_handcrafted", type=str, default="False", choices=["True", "False"])
    parser.add_argument("--model", type=str, required=True,
                         help="OpenRouter model ID used for repair/regeneration/verification.")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0,
                         help="Only process the first N files (0 = all). Useful for a cheap smoke test.")
    parser.add_argument("--output_tag", type=str, default="",
                         help="Suffix for output filenames (defaults to --pivot_source, plus '_gtreason' when --use_gt_reason is set).")
    parser.add_argument(
        "--use_gt_reason", action="store_true",
        help="Off by default (recommended): the repair step is only told a step was "
             "flagged as decisive and must self-diagnose it, so ground_truth never "
             "enters the repair/regeneration pipeline at all. When set, an explicit "
             "ground-truth-informed diagnosis is generated first and passed through a "
             "leak gate (literal check + LLM judge + retries + GT-free fallback) before "
             "being used -- a secondary ablation, not the recommended headline setting."
    )
    args = parser.parse_args()

    client, base_url = build_client(args)
    is_handcrafted = args.is_handcrafted == "True"
    index_agent = "role" if is_handcrafted else "name"

    if args.pivot_source == "attribution_log":
        if not args.attribution_log:
            print("Error: --attribution_log is required when --pivot_source=attribution_log")
            sys.exit(1)
        pivots = read_pivot_predictions(args.attribution_log)
    else:
        pivots = {}

    json_files = _get_sorted_json_files(args.directory_path)
    if args.limit and args.limit > 0:
        json_files = json_files[: args.limit]

    output_dir = "outputs_repair"
    os.makedirs(output_dir, exist_ok=True)
    tag = args.output_tag or (args.pivot_source + ("_gtreason" if args.use_gt_reason else ""))
    model_slug = args.model.replace("/", "_")
    log_path = os.path.join(output_dir, f"repair_{model_slug}_{tag}.txt")
    results_path = os.path.join(output_dir, f"results_{model_slug}_{tag}.jsonl")

    print(f"Pivot source: {args.pivot_source}")
    print(f"Reason mode: {'GT-informed (gated)' if args.use_gt_reason else 'GT-free (default)'}")
    print(f"Model: {args.model}")
    print(f"Files to process: {len(json_files)}")
    print(f"Log: {log_path}")
    print(f"Results: {results_path}")

    n_attempted = 0
    n_success = 0

    with open(log_path, "w", encoding="utf-8") as log_file, \
         open(results_path, "w", encoding="utf-8") as results_file:

        with contextlib.redirect_stdout(log_file):
            print("--- Trajectory Repair Run ---")
            print(f"Timestamp: {datetime.datetime.now()}")
            print(f"Model: {args.model}  Base URL: {base_url}")
            print(f"Pivot source: {args.pivot_source}")
            print(f"Reason mode: {'GT-informed (gated)' if args.use_gt_reason else 'GT-free (default)'}")
            print(f"Data: {args.directory_path}  (is_handcrafted={is_handcrafted})")
            print("-" * 20)

        for json_file in tqdm(json_files, desc="Repairing"):
            file_path = os.path.join(args.directory_path, json_file)
            data = _load_json_data(file_path)
            if not data:
                continue
            chat_history = data.get("history", [])
            query = data.get("question", "")
            ground_truth = data.get("ground_truth", "")
            if not chat_history:
                continue

            # --- resolve s* (decisive error step) and its agent ---
            if args.pivot_source == "attribution_log":
                pred = pivots.get(json_file)
                if not pred:
                    with contextlib.redirect_stdout(log_file):
                        print(f"[SKIP] {json_file}: no attribution prediction found.")
                    continue
                try:
                    s_star = int(pred["predicted_step"])
                except (KeyError, ValueError):
                    with contextlib.redirect_stdout(log_file):
                        print(f"[SKIP] {json_file}: unparsable predicted step.")
                    continue
                predicted_agent = pred.get("predicted_agent", "")
            elif args.pivot_source == "oracle":
                oracle = get_oracle_pivot(data)
                if oracle is None:
                    with contextlib.redirect_stdout(log_file):
                        print(f"[SKIP] {json_file}: no oracle label.")
                    continue
                s_star, predicted_agent = oracle
            else:  # random
                rnd = get_random_pivot(data, json_file, index_agent)
                if rnd is None:
                    with contextlib.redirect_stdout(log_file):
                        print(f"[SKIP] {json_file}: empty history.")
                    continue
                s_star, predicted_agent = rnd

            if not (0 <= s_star < len(chat_history)):
                with contextlib.redirect_stdout(log_file):
                    print(f"[SKIP] {json_file}: s*={s_star} out of range (len={len(chat_history)}).")
                continue

            # Was s* the same step the dataset's own label says is the decisive
            # error? (independent of pivot_source -- lets us ask "does repair
            # rate differ when the attributed location happens to be right vs.
            # wrong", not just "which pivot_source did we use".) This uses the
            # label only for bucketing results after the fact; it never
            # influences the repair itself.
            mistake_step_raw = data.get("mistake_step")
            mistake_agent_raw = data.get("mistake_agent")
            correct_attribution = attribution_is_correct(s_star, mistake_step_raw)

            n_attempted += 1

            with contextlib.redirect_stdout(log_file):
                print(f"\n--- Repairing {json_file} (s*={s_star}, agent={predicted_agent}, "
                      f"attribution_correct={correct_attribution}) ---")

            if args.use_gt_reason:
                # Opt-in only: ground_truth is used here, but the result passes
                # through a hard leak gate before repair_step ever sees it.
                reason = explain_failure_without_leaking_answer(
                    client, args.model, args.max_tokens, query, ground_truth, chat_history, s_star, index_agent
                )
            else:
                # Default: no ground_truth anywhere in the repair/regeneration path.
                reason = None
            corrected = repair_step(
                client, args.model, args.max_tokens, query, chat_history, s_star, index_agent, reason
            )
            downstream = regenerate_downstream(
                client, args.model, args.max_tokens, query, chat_history, s_star, index_agent, corrected
            )
            final_output = "\n".join(
                f"{entry.get(index_agent)}: {entry.get('content')}" for entry in downstream
            )
            success, verdict = verify_success(client, args.model, args.max_tokens, query, ground_truth, final_output)

            with contextlib.redirect_stdout(log_file):
                print(f"Reason given to repair_step: {reason if reason else '(none -- GT-free mode, self-diagnosed)'}")
                print(f"Corrected step {s_star} output:\n{corrected}")
                for k, entry in enumerate(downstream):
                    print(f"Regenerated step {s_star + k} - {entry.get(index_agent)}: {entry.get('content')}")
                print(f"Verifier: {verdict}")
                print(f"=== Repair Result for {json_file}: {'SUCCESS' if success else 'FAIL'} ===")

            if success:
                n_success += 1

            results_file.write(json.dumps({
                "file": json_file,
                "pivot_source": args.pivot_source,
                "use_gt_reason": args.use_gt_reason,
                "s_star": s_star,
                "predicted_agent": predicted_agent,
                "mistake_step": mistake_step_raw,
                "mistake_agent": mistake_agent_raw,
                "attribution_correct": correct_attribution,
                "success": success,
                "verifier_verdict": verdict,
            }, ensure_ascii=False) + "\n")
            results_file.flush()

        with contextlib.redirect_stdout(log_file):
            print("-" * 20)
            print(f"Attempted: {n_attempted}  Success: {n_success}")
            if n_attempted:
                print(f"Recovery rate: {n_success / n_attempted * 100:.2f}%")

    print(f"Done. Attempted: {n_attempted}  Success: {n_success}"
          + (f"  Recovery rate: {n_success / n_attempted * 100:.2f}%" if n_attempted else ""))
    print(f"Log: {log_path}")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
