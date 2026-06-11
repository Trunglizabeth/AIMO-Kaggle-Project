"""CLI wrapper to run the evaluation script with convenient flags.

Examples:
    python scripts/run_evaluation.py --csv data/train.csv --report data/eval.csv --limit 10 --use-vllm --use-majority-voting

The script sets environment variables `USE_VLLM` and `USE_MAJORITY_VOTING` for compatibility.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is importable when running from scripts/
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts import evaluate as eval_module


def parse_args():
    p = argparse.ArgumentParser(description='Run evaluation with options')
    p.add_argument('--csv', default='data/train.csv', help='Input CSV path')
    p.add_argument('--report', default='data/evaluation_report.csv', help='Output report CSV path')
    p.add_argument('--limit', type=int, default=10, help='Number of examples to evaluate')
    p.add_argument('--use-vllm', action='store_true', help='Use local vLLM engine')
    p.add_argument('--use-majority-voting', action='store_true', help='Use majority voting self-consistency')
    p.add_argument('--allow-mock', action='store_true', help='Allow mock LLM fallback when API is unavailable')
    p.add_argument('--timeout', type=int, default=10, help='Execution timeout in seconds')
    p.add_argument('--debug', action='store_true', help='Print debug output while evaluating')
    p.add_argument('--max-examples', type=int, default=None, help='Optional maximum number of examples to evaluate')
    return p.parse_args()


def main():
    args = parse_args()

    os.environ['USE_VLLM'] = 'true' if args.use_vllm else 'false'
    os.environ['USE_MAJORITY_VOTING'] = 'true' if args.use_majority_voting else 'false'
    if args.allow_mock:
        os.environ['ALLOW_MOCK_LLM'] = 'true'

    print(f"Running evaluation: csv={args.csv}, report={args.report}, limit={args.limit}, use_vllm={args.use_vllm}, use_majority_voting={args.use_majority_voting}, timeout={args.timeout}, debug={args.debug}")

    eval_module.evaluate(
        csv_path=args.csv,
        report_path=args.report,
        limit=args.limit if args.max_examples is None else min(args.limit, args.max_examples),
        use_majority_voting=args.use_majority_voting,
        use_vllm=args.use_vllm,
        timeout=args.timeout,
        debug=args.debug,
    )


if __name__ == '__main__':
    main()
