"""Evaluate the AIMO PoT pipeline on a dataset.

Reads `data/train.csv` with columns ['id','problem','answer'] and runs the pipeline
(LLM -> parser -> executor) on the first 10 rows. Produces `data/evaluation_report.csv`
with per-example results and prints overall accuracy.
"""

import os
import pandas as pd
import time
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


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


def evaluate(csv_path: str = 'data/train.csv', report_path: str = 'data/evaluation_report.csv', limit: int = 10):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[['id', 'problem', 'answer']].head(limit)

    engine = None
    try:
        engine = LLMEngineAPI()
    except Exception as e:
        if os.getenv('ALLOW_MOCK_LLM', 'false').lower() in ('1', 'true', 'yes'):
            print("Cảnh báo: Không thể khởi tạo API thật, đang dùng Mock Engine.")
            engine = LLMEngineAPI(mock_mode=True) # Nên thiết kế class hỗ trợ tham số này
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

        # Step A: get raw LLM output
        raw = engine.generate_code(problem)

        # Step B: extract python code
        code = ResponseParser.extract_python_code(raw)
        # Step B: extract python code
        code = ResponseParser.extract_python_code(raw)
        
        # --- THÊM 3 DÒNG DEBUG NÀY VÀO ---
        if example_id == "9622deee": # ID của bài toán Đếm số (Row 5)
            print("\n" + "="*40 + " DEBUG CODE " + "="*40)
            print(code)
            print("="*92 + "\n")
        # --------------------------------
        if not code:
            error_log = 'No code extracted from LLM output.'
            results.append({
                'id': example_id,
                'problem': problem,
                'true_answer': true_answer_raw,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'error_log': error_log,
            })
            continue

        # Step C: execute code
        out = executor.execute_code(code, timeout=5)

        # Check for errors from executor
        if out == 'Timeout Error' or (isinstance(out, str) and out.startswith('Error')):
            error_log = out
            predicted_answer = None
        else:
            try:
                pred_int = safe_int(out)
                predicted_answer = pred_int
                if true_answer is not None and pred_int is not None:
                    # SỬA Ở ĐÂY: Thêm logic Modulo 1000 chuẩn AIMO
                    is_correct = ((int(pred_int) % 1000) == (int(true_answer) % 1000))
                else:
                    is_correct = str(predicted_answer) == str(true_answer_raw)
            except Exception as e:
                error_log = f'Parsing predicted answer failed: {e}'
                predicted_answer = None

        results.append({
            'id': example_id,
            'problem': problem,
            'true_answer': true_answer_raw,
            'predicted_answer': predicted_answer,
            'is_correct': is_correct,
            'error_log': error_log,
        })
        time.sleep(2)

    report_df = pd.DataFrame(results, columns=['id', 'problem', 'true_answer', 'predicted_answer', 'is_correct', 'error_log'])

    os.makedirs(os.path.dirname(report_path) or '.', exist_ok=True)
    report_df.to_csv(report_path, index=False)

    # Print overall accuracy
    total = len(report_df)
    correct = int(report_df['is_correct'].sum())
    accuracy = correct / total if total > 0 else 0.0
    print(f'Pass@1 Accuracy: {accuracy:.2%} ({correct}/{total})')


if __name__ == '__main__':
    evaluate()
