"""Configuration module for AIMO Solver Pipeline."""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration class for AIMO Program-of-Thought (PoT) solver."""

    # LLM Configuration
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    MODEL_NAME: str = "gemini-1.5-pro-latest"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 2048

    # Model path (Kaggle)
    MODEL_PATH: str = "/kaggle/input/qwen-math-awq/qwen2.5-math-7b-instruct-awq"
    GPU_MEMORY_UTILIZATION: float = 0.9

    # Inference Configuration
    MAJORITY_VOTING_N: int = 15
    EXECUTION_TIMEOUT_SECONDS: int = 5

    # Data Paths
    TRAIN_CSV: str = "data/raw/train.csv"
    VALIDATION_CSV: str = "data/validation/validation.csv"
    REPORT_CSV: str = "scripts/report.csv"

    # Logging
    WANDB_PROJECT: str = "AIMO-Kaggle"
    WANDB_MODE: str = "offline"

    def __post_init__(self):
        pass

    def validate(self, require_api: bool = False):
        if require_api and not self.LLM_API_KEY:
            raise ValueError(
                "LLM_API_KEY environment variable not set. "
                "Please set it before running the pipeline."
            )


# Create a default config instance
config = Config()