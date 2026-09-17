import os
import argparse
import contextlib
import sys
import datetime
from dotenv import load_dotenv
from openai import OpenAI

from Lib.utils import (
    all_at_once as gpt_all_at_once,
    step_by_step as gpt_step_by_step,
    all_at_once_with_window as gpt_all_at_once_2stage,
    step_by_step_with_window as gpt_step_by_step_2stage,
)

OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Analyze multi-agent chat history using an OpenRouter-hosted LLM."
    )
    parser.add_argument(
        "--two_stage_window",
        action="store_true",
        help="Use 2-stage judging: (1) global/step-by-step prediction, then (2) window-based final judge around the predicted step.",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["all_at_once", "step_by_step"],
        help="The analysis method to use."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "OpenRouter model ID, e.g. 'openai/gpt-4o', 'openai/gpt-4o-mini', "
            "'anthropic/claude-3.5-sonnet', 'meta-llama/llama-3.1-70b-instruct'. "
            "Any model id listed at https://openrouter.ai/models works."
        )
    )
    parser.add_argument(
        "--directory_path",
        type=str,
        default="../Who&When/Algorithm-Generated",
        help="Path to the directory containing JSON chat history files. Default: '../Who&When/Algorithm-Generated'."
    )

    parser.add_argument(
        "--is_handcrafted",
        type=str,
        default="False",
        choices=['True', 'False'], # If you want to test Hand-Crafted, set is_handcrafted to be True.
        help="Specify 'True' or 'False'. Default: 'False'."
    )

    # OpenRouter API 키 (.env 의 OPENROUTER_API_KEY 를 기본값으로 사용)
    parser.add_argument(
        "--api_key",
        type=str,
        default="",  # 공백 문자열이 아닌 진짜 빈 문자열로 두어 검증이 제대로 일어나도록
        help="OpenRouter API key. If omitted, uses the OPENROUTER_API_KEY environment variable (.env)."
    )

    parser.add_argument(
        "--base_url",
        type=str,
        default="",
        help=(
            "OpenAI-compatible API base URL. If omitted, uses OPENROUTER_BASE_URL from .env, "
            f"else defaults to '{OPENROUTER_DEFAULT_BASE_URL}'."
        )
    )

    parser.add_argument(
        "--max_tokens",  # utils 내부가 chat/completions 를 호출할 때 그대로 전달만 함
        type=int,
        default=1024,
        help="Maximum number of tokens for the LLM response (forwarded to utils)."
    )

    args = parser.parse_args()

    # CLI 키 우선, 없으면 env(OPENROUTER_API_KEY) 사용
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

    # OpenRouter 랭킹에 앱을 표시하기 위한 선택적 헤더 (https://openrouter.ai/docs)
    default_headers = {}
    site_url = (os.getenv("OPENROUTER_SITE_URL") or "").strip()
    site_name = (os.getenv("OPENROUTER_SITE_NAME") or "").strip()
    if site_url:
        default_headers["HTTP-Referer"] = site_url
    if site_name:
        default_headers["X-Title"] = site_name

    try:
        client = OpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers or None)
        print(f"OpenRouter client initialized (base_url={base_url}).")
    except Exception as e:
        print(f"Error initializing OpenRouter client: {e}")
        sys.exit(1)

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    handcrafted_suffix = "_handcrafted" if args.is_handcrafted == "True" else "_alg_generated"
    model_slug = args.model.replace('/', '_')
    output_filename = f"{args.method}_{model_slug}{handcrafted_suffix}.txt"
    output_filepath = os.path.join(output_dir, output_filename)

    args.is_handcrafted = True if args.is_handcrafted == "True" else False # Update: Convert string to boolean

    print(f"Analysis method: {args.method}")
    print(f"Model (OpenRouter): {args.model}")
    print(f"Output will be saved to: {output_filepath}")

    try:
        with open(output_filepath, 'w', encoding='utf-8') as output_file, contextlib.redirect_stdout(output_file):
            print(f"--- Starting Analysis: {args.method} ---")
            print(f"Timestamp: {datetime.datetime.now()}")
            print(f"Model Used (OpenRouter): {args.model}")
            print(f"Input Directory: {args.directory_path}")
            print(f"Is Handcrafted: {args.is_handcrafted}")
            print("-" * 20)

            if args.method == "all_at_once":
                if args.two_stage_window:
                    gpt_all_at_once_2stage(
                        client=client,
                        directory_path=args.directory_path,
                        is_handcrafted=args.is_handcrafted,
                        model=args.model,
                        max_tokens=args.max_tokens,
                    )
                else:
                    gpt_all_at_once(
                        client=client,
                        directory_path=args.directory_path,
                        is_handcrafted=args.is_handcrafted,
                        model=args.model,
                        max_tokens=args.max_tokens,
                    )
            elif args.method == "step_by_step":
                if args.two_stage_window:
                    gpt_step_by_step_2stage(
                        client=client,
                        directory_path=args.directory_path,
                        is_handcrafted=args.is_handcrafted,
                        model=args.model,
                        max_tokens=args.max_tokens,
                    )
                else:
                    gpt_step_by_step(
                        client=client,
                        directory_path=args.directory_path,
                        is_handcrafted=args.is_handcrafted,
                        model=args.model,
                        max_tokens=args.max_tokens,
                    )

            print("-" * 20)
            print(f"--- Analysis Complete ---")

        print(f"Analysis finished. Output saved to {output_filepath}")

    except Exception as e:
        print(f"\n!!! An error occurred during analysis or file writing: {e} !!!", file=sys.stderr)

if __name__ == "__main__":
    main()
