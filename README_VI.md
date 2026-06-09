# AIMO-Kaggle-Project

Pipeline giải toán Olympic Toán học AI (AIMO) theo hướng Program-of-Thought (PoT).

Khoản này cung cấp PoC pipeline cho các bước:

- Gửi đề toán cho LLM (OpenAI hoặc Google GenAI)
- Trích xuất mã Python từ phản hồi của LLM
- Thực thi mã Python trong sandbox an toàn

## Hướng dẫn nhanh

1. Cài phụ thuộc:

```bash
pip install -r requirements.txt
```

2. (Tuỳ chọn) Thiết lập API key LLM:

PowerShell (Windows):

```powershell
$env:LLM_API_KEY = "your_api_key"
```

3. Chạy PoC runner:

```bash
python main_poc.py
```

## Chuẩn bị dữ liệu mẫu

Nếu bạn muốn tạo dữ liệu AIME mẫu từ Hugging Face:

```bash
python scripts/prepare_data.py
```

Script này tải bộ dữ liệu `gneubig/aime-1983-2024`, lấy 50 bài mẫu và lưu vào `data/train.csv`.

## Chạy local vLLM

```powershell
$env:USE_VLLM = "true"
python main_poc.py
```

## Chạy majority voting evaluation

```powershell
$env:USE_VLLM = "true"
python scripts/evaluate.py --use-majority-voting --use-vllm --timeout 10 --limit 10
```

Hoặc dùng wrapper CLI:

```powershell
python scripts/run_evaluation.py --csv data/train.csv --report data/evaluation_report.csv --limit 10 --use-vllm --use-majority-voting --timeout 10 --debug
```

## Hỗ trợ Kaggle offline

Tạo script cài đặt wheel offline:

```powershell
python scripts/prepare_kaggle_offline.py --dataset-dir /kaggle/input/aimo-offline-packages
```

Kiểm tra wheel packages trước khi cài đặt:

```powershell
python scripts/verify_offline_packages.py --dataset-dir /kaggle/input/aimo-offline-packages
```

## Tạo submission Kaggle

```powershell
python scripts/generate_submission.py --csv data/train.csv --output submission.csv --use-vllm --use-majority-voting --allow-mock
```

## Triển khai trên Kaggle

```bash
python scripts/deploy_to_kaggle.py \
  --dataset-dir /kaggle/input/aimo-offline-packages \
  --csv data/train.csv \
  --output submission.csv \
  --use-vllm \
  --use-majority-voting \
  --allow-mock
```

## Phân tích đánh giá (Sprint 3)

Sau khi có `data/evaluation_report.csv`, chạy:

```bash
python scripts/analyze_evaluation.py --report data/evaluation_report.csv --output-summary data/report_summary.json --output-markdown data/report_summary.md
```

Hoặc chạy end-to-end đánh giá và phân tích:

```bash
python scripts/evaluate_and_analyze.py --csv data/train.csv --report data/evaluation_report.csv --use-vllm --use-majority-voting --output-summary data/report_summary.json --output-markdown data/report_summary.md
```

Script này in:

- tổng số mẫu
- số dự đoán đúng
- độ chính xác
- số mẫu không có dự đoán
- số lỗi timeout
- top lỗi xuất hiện nhiều nhất

## Các tệp chính

- `src/engine_api.py`: wrapper LLM với `LLMEngineAPI` và `SYSTEM_PROMPT`.
- `src/parser.py`: `ResponseParser` để trích xuất mã Python.
- `src/executor.py`: `PythonExecutor` để thực thi mã an toàn.
- `main_poc.py`: chạy pipeline end-to-end.
- `scripts/run_evaluation.py`: CLI chạy đánh giá.
- `scripts/generate_submission.py`: tạo submission Kaggle.
- `scripts/deploy_to_kaggle.py`: deploy và cài offline wheel trên Kaggle.
- `scripts/analyze_evaluation.py`: phân tích báo cáo đánh giá.

## Testing

Chạy unit test:

```bash
pytest -q
```

## Sprint 2

Sprint 2 đã hoàn thành với:

- hỗ trợ local vLLM
- majority voting
- evaluation CLI
- submission generation
- hỗ trợ offline Kaggle

## Sprint 3

Sprint 3 đã hoàn thiện với:

- phân tích báo cáo đánh giá bằng `scripts/analyze_evaluation.py`
- chạy end-to-end đánh giá + phân tích bằng `scripts/evaluate_and_analyze.py`
- mở rộng chuẩn bị dữ liệu bằng `scripts/prepare_data.py`
- hoàn thiện luồng Kaggle end-to-end bằng `scripts/generate_submission.py`, `scripts/deploy_to_kaggle.py`, `scripts/verify_offline_packages.py`, và `scripts/prepare_kaggle_offline.py`

## Sprint 3

Sprint 3 đã hoàn thiện với:

- phân tích báo cáo đánh giá bằng `scripts/analyze_evaluation.py`
- chạy end-to-end đánh giá + phân tích bằng `scripts/evaluate_and_analyze.py`
- mở rộng chuẩn bị dữ liệu bằng `scripts/prepare_data.py`
- hoàn thiện luồng Kaggle end-to-end bằng `scripts/generate_submission.py`, `scripts/deploy_to_kaggle.py`, `scripts/verify_offline_packages.py`, và `scripts/prepare_kaggle_offline.py`
