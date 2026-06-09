"""Generate offline Kaggle installation helper script for vLLM and math dependencies.

This helper script is intended to create a shell script that installs pre-uploaded
wheel files from a Kaggle Dataset. The actual wheels should be uploaded to a Kaggle
Dataset first, then this script can generate the commands needed to install them.

Usage:
    python scripts/prepare_kaggle_offline.py --dataset-dir /kaggle/input/aimo-offline-packages
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


DEFAULT_WHEEL_FILES = [
    'vllm-*.whl',
    'xformers-*.whl',
    'sympy-*.whl',
    'torch-*.whl',
    'transformers-*.whl',
]


def build_install_commands(dataset_dir: str, wheel_files: List[str]) -> List[str]:
    commands = [
        '#!/bin/bash',
        'set -euo pipefail',
        '',
        '# Install offline wheels from the Kaggle dataset',
    ]

    for wheel_pattern in wheel_files:
        commands.append(f'echo "Installing {wheel_pattern}..."')
        commands.append(
            f'pip install --no-index --find-links "{dataset_dir}" "{wheel_pattern}"'
        )

    commands.append('echo "Offline installation complete."')
    return commands


def save_script(lines: List[str], output_path: Path) -> None:
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    output_path.chmod(0o755)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare Kaggle offline install script for AIMO project.')
    parser.add_argument(
        '--dataset-dir',
        default='/kaggle/input/aimo-offline-packages',
        help='Path to the Kaggle dataset folder that contains wheel files.',
    )
    parser.add_argument(
        '--output',
        default='install_offline_dependencies.sh',
        help='Output shell script name.',
    )
    parser.add_argument(
        '--wheels',
        nargs='*',
        default=DEFAULT_WHEEL_FILES,
        help='Wheel filename patterns to install.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = build_install_commands(args.dataset_dir, args.wheels)
    output_path = Path(__file__).resolve().parent / args.output
    save_script(commands, output_path)
    print(f'Generated offline install script: {output_path}')
    print('Upload the required wheel files to the Kaggle Dataset and then run the script inside Kaggle.')


if __name__ == '__main__':
    main()
