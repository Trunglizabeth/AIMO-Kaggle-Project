"""Verify that required offline wheel packages exist inside a Kaggle dataset folder."""

from __future__ import annotations

import argparse
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
    result: dict[str, List[Path]] = {}
    for pattern in patterns:
        matches = sorted(dataset_dir.glob(pattern))
        result[pattern] = matches
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Verify required offline wheel packages in a Kaggle dataset directory.')
    parser.add_argument('--dataset-dir', default='/kaggle/input/aimo-offline-packages', help='Path to the Kaggle dataset containing wheel files.')
    parser.add_argument('--wheels', nargs='*', default=DEFAULT_WHEEL_PATTERNS, help='Optional list of wheel filename patterns to verify.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f'Dataset directory not found: {dataset_dir}')

    found = find_wheel_files(dataset_dir, args.wheels)
    missing = []

    print(f'Checking offline wheel directory: {dataset_dir}')
    for pattern, matches in found.items():
        if matches:
            print(f'  {pattern}: {len(matches)} file(s) found')
            for path in matches:
                print(f'    - {path.name}')
        else:
            print(f'  {pattern}: MISSING')
            missing.append(pattern)

    if missing:
        print('\nMissing required wheel patterns:')
        for pattern in missing:
            print(f'  - {pattern}')
        raise SystemExit(1)

    print('\nAll required wheel patterns are present.')


if __name__ == '__main__':
    main()
