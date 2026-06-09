"""Evaluate the AIMO PoT pipeline on a dataset.

Reads `data/train.csv` with columns ['id','problem','answer'] and runs the pipeline
(LLM -> parser -> executor) on a limited number of rows. Produces
`data/evaluation_report.csv` with per-example results and prints overall accuracy.
"""

import argparse
import os
import pandas as pd
import time
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


from src.config import config
from src.engine_api import LLMEngineAPI
from src.parser import ResponseParser
from src.executor import PythonExecutor


def safe_int(x):
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return None


def evaluate(
    csv_path: str = 'data/train.csv',
    report_path: str = 'data/evaluation_report.csv',
    limit: int = 10,
    use_majority_voting: bool = False,
    use_vllm: bool = False,
    timeout: int = config.EXECUTION_TIMEOUT_SECONDS,
    debug: bool = False,
) -> None:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[['id', 'problem', 'answer']].head(limit)

    engine = None
    try:
        engine = LLMEngineAPI(use_vllm=use_vllm)
    except Exception as e:
        if os.getenv('ALLOW_MOCK_LLM', 'false').lower() in ('1', 'true', 'yes'):
            print("Cảnh báo: Không thể khởi tạo API thật, đang dùng Mock Engine.")
            engine = LLMEngineAPI(mock_mode=True)
        else:
            raise RuntimeError(f"Lỗi khởi tạo LLM API: {e}")

    parser = ResponseParser()
    executor = PythonExecutor()
    results = []

    for _, row in df.iterrows():
        example_id = row['id']
        problem = row['problem']
        true_answer_raw = row['answer']
        true_answer = safe_int(true_answer_raw)

        error_log = ''
        predicted_answer = None
        is_correct = False

        if use_majority_voting:
            predicted_answer = engine.solve_with_majority_voting(
                problem,
                n=config.MAJORITY_VOTING_N,
                timeout=timeout,
            )
            try:
                if true_answer is not None and predicted_answer is not None:
                    is_correct = ((int(predicted_answer) % 1000) == (int(true_answer) % 1000))
                else:
                    is_correct = str(predicted_answer) == str(true_answer_raw)
            except Exception:
                is_correct = False

            results.append({
                'id': example_id,
                'problem': problem,
                'true_answer': true_answer_raw,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'error_log': error_log,
            })
            continue

        raw = engine.generate_code(problem)
        code = ResponseParser.extract_python_code(raw)
        prediction_method = 'code'

        if debug:
            print(f'--- Example {example_id} ---')
            print(f'Problem: {problem}')
            print(f'Raw LLM output:\n{raw}')
            print(f'Extracted code:\n{code}')

        if not code:
            raw_answer = ResponseParser.extract_numeric_answer(raw)
            if raw_answer:
                predicted_answer = safe_int(raw_answer) or raw_answer
                prediction_method = 'raw_answer'
                try:
                    if true_answer is not None and predicted_answer is not None:
                        is_correct = ((int(predicted_answer) % 1000) == (int(true_answer) % 1000))
                    else:
                        is_correct = str(predicted_answer) == str(true_answer_raw)
                except Exception:
                    is_correct = False
            else:
                error_log = 'No code extracted from LLM output.'
                prediction_method = 'no_code'

            results.append({
                'id': example_id,
                'problem': problem,
                'true_answer': true_answer_raw,
                'predicted_answer': predicted_answer,
                'prediction_method': prediction_method,
                'is_correct': is_correct,
                'error_log': error_log,
            })
            continue

        out = executor.execute_code(code, timeout=timeout)

        if out == 'Timeout Error' or (isinstance(out, str) and out.startswith('Error')):
            error_log = out
            predicted_answer = None
        else:
            try:
                pred_int = safe_int(out)
                predicted_answer = pred_int
                if true_answer is not None and pred_int is not None:
                    is_correct = ((int(pred_int) % 1000) == (int(true_answer) % 1000))
                else:
                    is_correct = str(predicted_answer) == str(true_answer_raw)
            except Exception as e:
                error_log = f'Parsing predicted answer failed: {e}'
                predicted_answer = None

        if debug:
            print(f'Execution output: {out}')
            print(f'Predicted answer: {predicted_answer}')
            print(f'Is correct: {is_correct}')
            print(f'Error log: {error_log}')

        results.append({
            'id': example_id,
            'problem': problem,
            'true_answer': true_answer_raw,
            'predicted_answer': predicted_answer,
            'prediction_method': prediction_method,
            'is_correct': is_correct,
            'error_log': error_log,
        })
        time.sleep(2)

    report_df = pd.DataFrame(results, columns=['id', 'problem', 'true_answer', 'predicted_answer', 'prediction_method', 'is_correct', 'error_log'])
    os.makedirs(os.path.dirname(report_path) or '.', exist_ok=True)
    report_df.to_csv(report_path, index=False)

    total = len(report_df)
    correct = int(report_df['is_correct'].sum())
    accuracy = correct / total if total > 0 else 0.0
    print(f'Pass@1 Accuracy: {accuracy:.2%} ({correct}/{total})')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate the AIMO PoT pipeline on a dataset.')
    parser.add_argument('--csv', default='data/train.csv', help='Input CSV path.')
    parser.add_argument('--report', default='data/evaluation_report.csv', help='Output report CSV path.')
    parser.add_argument('--limit', type=int, default=10, help='Limit number of examples to evaluate.')
    parser.add_argument('--use-vllm', action='store_true', help='Use local vLLM engine if available.')
    parser.add_argument('--use-majority-voting', action='store_true', help='Use majority voting self-consistency.')
    parser.add_argument('--allow-mock', action='store_true', help='Allow mock LLM when API is unavailable.')
    parser.add_argument('--timeout', type=int, default=config.EXECUTION_TIMEOUT_SECONDS, help='Execution timeout in seconds.')
    parser.add_argument('--debug', action='store_true', help='Print debug output for each example.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.allow_mock:
        os.environ['ALLOW_MOCK_LLM'] = 'true'

    evaluate(
        csv_path=args.csv,
        report_path=args.report,
        limit=args.limit,
        use_majority_voting=args.use_majority_voting,
        use_vllm=args.use_vllm,
        timeout=args.timeout,
        debug=args.debug,
    )


if __name__ == '__main__':
    main()
