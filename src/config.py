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

    # Inference Configuration
    MAJORITY_VOTING_N: int = 15
    EXECUTION_TIMEOUT_SECONDS: int = 5

    def __post_init__(self):
        """Validate configuration after initialization.

        By default, configuration should not fail on import because the project
        supports mock LLM execution when no API key is available.
        """
        pass

    def validate(self, require_api: bool = False):
        """Validate configuration explicitly.

        Args:
            require_api (bool): If True, raise an error when no LLM_API_KEY is set.
        """
        if require_api and not self.LLM_API_KEY:
            raise ValueError(
                "LLM_API_KEY environment variable not set. "
                "Please set it before running the pipeline."
            )


# Create a default config instance
config = Config()
