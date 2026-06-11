"""Deploy the AIMO solver on Kaggle by installing offline wheels and generating a submission.

This script is intended for Kaggle notebook or container environments where dependencies
are pre-downloaded as wheel files inside a dataset mount.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

DEFAULT_WHEEL_PATTERNS = [
    'vllm-*.whl',
    'xformers-*.whl',
    'sympy-*.whl',
    'torch-*.whl',
    'transformers-*.whl',
]


def find_wheel_files(dataset_dir: Path, patterns: List[str]) -> dict[str, List[Path]]:
    results: dict[str, List[Path]] = {}
    for pattern in patterns:
        results[pattern] = sorted(dataset_dir.glob(pattern))
    return results


def install_wheel_packages(dataset_dir: Path, patterns: List[str]) -> None:
    found = find_wheel_files(dataset_dir, patterns)
    missing = [pattern for pattern, paths in found.items() if not paths]
    if missing:
        raise FileNotFoundError(f'Missing required wheel patterns: {missing}')

    wheel_paths = [str(path) for paths in found.values() for path in paths]
    command = [sys.executable, '-m', 'pip', 'install', '--no-index', '--find-links', str(dataset_dir), *wheel_paths]
    print('Installing offline wheel packages...')
    print(' '.join(command))
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Deploy AIMO solver inside Kaggle with offline wheels and submission generation.')
    parser.add_argument('--dataset-dir', default='/kaggle/input/aimo-offline-packages', help='Path to Kaggle dataset containing wheel files.')
    parser.add_argument('--csv', default='data/train.csv', help='Input CSV file for predictions.')
    parser.add_argument('--output', default='submission.csv', help='Output submission CSV file.')
    parser.add_argument('--use-vllm', action='store_true', help='Use local vLLM engine for generation.')
    parser.add_argument('--use-majority-voting', action='store_true', help='Use majority voting for predictions.')
    parser.add_argument('--allow-mock', action='store_true', help='Allow mock fallback when LLM API is unavailable.')
    parser.add_argument('--wheel-patterns', nargs='*', default=DEFAULT_WHEEL_PATTERNS, help='Optional list of wheel filename patterns to verify and install.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f'Dataset directory does not exist: {dataset_dir}')

    install_wheel_packages(dataset_dir, args.wheel_patterns)

    env = os.environ.copy()
    env['USE_VLLM'] = 'true' if args.use_vllm else 'false'
    env['USE_MAJORITY_VOTING'] = 'true' if args.use_majority_voting else 'false'
    if args.allow_mock:
        env['ALLOW_MOCK_LLM'] = 'true'

    command = [sys.executable, str(Path(__file__).resolve().parent / 'generate_submission.py'), '--csv', args.csv, '--output', args.output]
    if args.use_vllm:
        command.append('--use-vllm')
    if args.use_majority_voting:
        command.append('--use-majority-voting')
    if args.allow_mock:
        command.append('--allow-mock')

    print('Generating submission file...')
    subprocess.run(command, env=env, check=True)
    print(f'Submission generated at {args.output}')


if __name__ == '__main__':
    main()
