"""Generate a Kaggle-style prediction CSV from the AIMO dataset.

This script reads a CSV with columns ['id', 'problem', 'answer'] and writes a
submission CSV with ['id', 'prediction'] based on the current LLM pipeline.

Usage:
    python scripts/generate_submission.py --csv data/train.csv --output submission.csv --use-vllm --use-majority-voting
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.config import config
from src.engine_api import LLMEngineAPI
from src.parser import ResponseParser
from src.executor import PythonExecutor


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate Kaggle-style predictions from AIMO dataset.')
    parser.add_argument('--csv', default='data/train.csv', help='Input CSV path containing id, problem, answer columns.')
    parser.add_argument('--output', default='submission.csv', help='Output prediction CSV path.')
    parser.add_argument('--limit', type=int, default=None, help='Optional limit on number of examples to predict.')
    parser.add_argument('--use-vllm', action='store_true', help='Use local vLLM engine if available.')
    parser.add_argument('--use-majority-voting', action='store_true', help='Use majority voting self-consistency for predictions.')
    parser.add_argument('--allow-mock', action='store_true', help='Allow mock LLM fallback when API is unavailable.')
    return parser.parse_args(argv)


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


def predict_problem(engine: LLMEngineAPI, parser: ResponseParser, executor: PythonExecutor, problem_text: str, use_majority_voting: bool, timeout: int) -> int | None:
    if use_majority_voting:
        try:
            return engine.solve_with_majority_voting(problem_text, n=config.MAJORITY_VOTING_N, timeout=timeout)
        except Exception:
            return None

    try:
        raw = engine.generate_code(problem_text)
    except Exception:
        return None

    code = parser.extract_python_code(raw)
    if not code:
        return None

    result = executor.execute_code(code, timeout=timeout)
    if result == 'Timeout Error' or (isinstance(result, str) and result.startswith('Error')):
        return None

    return safe_int(result)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.environ['USE_VLLM'] = 'true' if args.use_vllm else 'false'
    if args.allow_mock:
        os.environ['ALLOW_MOCK_LLM'] = 'true'

    if not Path(args.csv).exists():
        raise FileNotFoundError(f'Input CSV file not found: {args.csv}')

    df = pd.read_csv(args.csv)
    if args.limit is not None:
        df = df.head(args.limit)

    engine = LLMEngineAPI(use_vllm=args.use_vllm)
    parser = ResponseParser()
    executor = PythonExecutor()

    predictions = []
    for _, row in df.iterrows():
        problem = row['problem']
        prediction = predict_problem(
            engine,
            parser,
            executor,
            problem,
            use_majority_voting=args.use_majority_voting,
            timeout=config.EXECUTION_TIMEOUT_SECONDS,
        )
        predictions.append({'id': row['id'], 'prediction': prediction if prediction is not None else ''})

    output_df = pd.DataFrame(predictions, columns=['id', 'prediction'])
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print(f'Generated submission file: {args.output}')


if __name__ == '__main__':
    main()
