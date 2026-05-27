# src/config.py

# =====================
# MODEL CONFIGURATION
# =====================
MODEL_PATH = "/kaggle/input/qwen-math-awq/qwen2.5-math-7b-instruct-awq"  # Đường dẫn model trên Kaggle
MAX_TOKENS = 1024          # Số token tối đa model sinh ra
GPU_MEMORY_UTILIZATION = 0.9  # Tận dụng 90% VRAM

# =====================
# SAMPLING CONFIGURATION
# =====================
N_SAMPLES = 15        # Số lần sinh code cho Majority Voting
TEMPERATURE = 0.7     # Độ ngẫu nhiên (0 = deterministic, 1 = random)
TIMEOUT_SECONDS = 5   # Thời gian tối đa chạy 1 đoạn code Python

# =====================
# DATA PATHS
# =====================
TRAIN_CSV = "data/raw/train.csv"
VALIDATION_CSV = "data/validation/validation.csv"
REPORT_CSV = "scripts/report.csv"

# =====================
# LOGGING
# =====================
WANDB_PROJECT = "AIMO-Kaggle"
WANDB_MODE = "offline"   # Chạy offline trên Kaggle, sync sau