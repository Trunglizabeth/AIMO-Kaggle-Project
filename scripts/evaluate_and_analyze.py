"""Run evaluation and analysis together as a Sprint 3 end-to-end helper.

This script evaluates a dataset, analyzes the generated report, and optionally
creates a Kaggle-style submission.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts import analyze_evaluation as analyze_module
from scripts import evaluate as evaluate_module
from scripts import generate_submission as generate_module


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate and analyze AIMO solver output in one command.')
    parser.add_argument('--csv', default='data/train.csv', help='Input dataset CSV path.')
    parser.add_argument('--report', default='data/evaluation_report.csv', help='Output evaluation report path.')
    parser.add_argument('--limit', type=int, default=10, help='Limit number of examples to evaluate.')
    parser.add_argument('--use-vllm', action='store_true', help='Use local vLLM engine.')
    parser.add_argument('--use-majority-voting', action='store_true', help='Use majority voting for evaluation.')
    parser.add_argument('--allow-mock', action='store_true', help='Allow mock LLM fallback if no API is available.')
    parser.add_argument('--timeout', type=int, default=10, help='Timeout for code execution in seconds.')
    parser.add_argument('--debug', action='store_true', help='Print debug information during evaluation.')
    parser.add_argument('--top-errors', type=int, default=10, help='Number of top error messages to include in the analysis.')
    parser.add_argument('--top-wrong', type=int, default=5, help='Number of wrong examples to display in the analysis.')
    parser.add_argument('--output-summary', default='data/report_summary.json', help='Optional path to save JSON summary.')
    parser.add_argument('--output-markdown', default='data/report_summary.md', help='Optional path to save Markdown summary.')
    parser.add_argument('--generate-submission', action='store_true', help='Generate a Kaggle submission after evaluation.')
    parser.add_argument('--submission-output', default='submission.csv', help='Output submission CSV path.')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.allow_mock:
        os.environ['ALLOW_MOCK_LLM'] = 'true'
    os.environ['USE_VLLM'] = 'true' if args.use_vllm else 'false'
    os.environ['USE_MAJORITY_VOTING'] = 'true' if args.use_majority_voting else 'false'

    evaluate_module.evaluate(
        csv_path=args.csv,
        report_path=args.report,
        limit=args.limit,
        use_majority_voting=args.use_majority_voting,
        use_vllm=args.use_vllm,
        timeout=args.timeout,
        debug=args.debug,
    )

    stats = analyze_module.summarize_report(
        Path(args.report),
        top_errors=args.top_errors,
        top_wrong=args.top_wrong,
    )

    if args.output_summary:
        output_summary_path = Path(args.output_summary)
        output_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with output_summary_path.open('w', encoding='utf-8') as f:
            import json

            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f'Analysis summary saved to {output_summary_path}')

    if args.output_markdown:
        analyze_module.save_markdown_summary(stats, Path(args.output_markdown))
        print(f'Markdown summary saved to {args.output_markdown}')

    if args.generate_submission:
        submission_args = [
            '--csv', args.csv,
            '--output', args.submission_output,
        ]
        if args.use_vllm:
            submission_args.append('--use-vllm')
        if args.use_majority_voting:
            submission_args.append('--use-majority-voting')
        if args.allow_mock:
            submission_args.append('--allow-mock')

        generate_module.main(submission_args)
        print(f'Submission generated at {args.submission_output}')

    print('Evaluation and analysis workflow complete.')


if __name__ == '__main__':
    main()
