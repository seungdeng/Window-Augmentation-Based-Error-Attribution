"""
Score a repair.py run: recovery/repair rate = success / attempted, from its
results_*.jsonl file. Also supports comparing several runs (e.g.
attribution_log vs. oracle vs. random) side by side.

Ablation: for each file, also breaks the results down by whether the pivot
step used for repair (s_star) matched the dataset's own decisive-error label
(mistake_step) -- "attribution correct" vs "attribution incorrect" -- so you
can see the repair rate when the proposed method localized the error exactly
right vs. when it didn't (pass --no_breakdown to suppress this).

Usage:
    python score_repair.py --results_file outputs_repair/results_openai_gpt-4o_attribution_log.jsonl
    python score_repair.py --results_file A.jsonl B.jsonl C.jsonl
    python score_repair.py --results_file A.jsonl --no_breakdown
"""

import json
import argparse
from collections import defaultdict


def load_results(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summarize(rows):
    total = len(rows)
    success = sum(1 for r in rows if r.get("success"))
    return total, success, (success / total * 100 if total else 0.0)


def print_row(label, total, success, rate, indent=""):
    print(f"{indent}{label:<50} {total:>9} {success:>7} {rate:>13.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Compute recovery/repair rate from repair.py results.")
    parser.add_argument("--results_file", type=str, nargs="+", required=True,
                         help="One or more results_*.jsonl files produced by repair.py.")
    parser.add_argument("--no_breakdown", action="store_true",
                         help="Suppress the attribution-correct vs. attribution-incorrect breakdown.")
    args = parser.parse_args()

    header_label = "File / Pivot Source / Attribution"
    print(f"{header_label:<50} {'Attempted':>9} {'Success':>7} {'Repair Rate':>14}")
    print("-" * 84)

    for path in args.results_file:
        rows = load_results(path)
        by_source = defaultdict(list)
        for r in rows:
            by_source[r.get("pivot_source", "unknown")].append(r)

        print(f"\n{path}")
        for source, source_rows in by_source.items():
            total, success, rate = summarize(source_rows)
            print_row(f"  [{source}] overall", total, success, rate)

            if args.no_breakdown:
                continue

            correct_rows = [r for r in source_rows if r.get("attribution_correct") is True]
            incorrect_rows = [r for r in source_rows if r.get("attribution_correct") is False]
            unknown_rows = [r for r in source_rows if r.get("attribution_correct") is None]

            if correct_rows:
                total_c, success_c, rate_c = summarize(correct_rows)
                print_row("attribution correct (s* == mistake_step)", total_c, success_c, rate_c, indent="    ")
            if incorrect_rows:
                total_w, success_w, rate_w = summarize(incorrect_rows)
                print_row("attribution incorrect (s* != mistake_step)", total_w, success_w, rate_w, indent="    ")
            if unknown_rows:
                total_u, success_u, rate_u = summarize(unknown_rows)
                print_row("attribution unknown (missing/unparsable label)", total_u, success_u, rate_u, indent="    ")


if __name__ == "__main__":
    main()
