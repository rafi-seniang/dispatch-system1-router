"""Typed settings, read from .env at the service root. Every value the service reads is declared here."""

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

SERVICE_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL_ENGINES = ("laya", "llm", "embedding")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=SERVICE_ROOT / ".env", extra="ignore")

    # Personal secrets file for HF_TOKEN, JEV_API_KEY and GEMINI_API_KEY (optional when env vars are passed directly)
    secrets_env_path: Path | None = None
    # Engine the triage app serves
    triage_engine: Literal["laya", "jev", "gemini", "llm", "embedding"] = "jev"
    # Jev, TypeSafe's hosted System One model. JEV_API_KEY comes from secrets_env_path or the environment
    jev_base_url: str = "https://api.typesafe.ai"
    jev_model: str = "jev-latest"
    jev_timeout_seconds: float = Field(default=30, gt=0)
    # Requests in flight at once when a batch goes to Jev (one text per request)
    jev_max_concurrent_requests: int = Field(default=8, gt=0)
    # Gemini Decision 1, Google's hosted System One model. GEMINI_API_KEY comes from secrets_env_path or the environment
    gemini_decision_model: str = "gemini-decision-1"
    gemini_timeout_seconds: float = Field(default=30, gt=0)
    gemini_max_concurrent_requests: int = Field(default=8, gt=0)
    # cuda or cpu (used by local PyTorch engines)
    device: str = "cuda"
    # Small instruct LLM read as a decision model through its next-token probabilities
    llm_model_id: str = "Qwen/Qwen3.5-2B"
    # Embedding model for the nearest-description baseline
    embedding_model_id: str = "google/embeddinggemma-2"
    # Softmax temperature that turns cosine similarities into probabilities
    embedding_temperature: float = 0.02
    # Below this department probability the app sends the message to a person
    review_probability_threshold: float = 0.7
    # At or above this expected urgency (0 to 2) the app escalates at once
    escalate_urgency_score: float = 1.5
    # Browser origins allowed to call the API: the lab frontend in development and in preview
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:4173"]
    environment: Literal["local", "test", "staging", "production"] = "local"

    # Redis for the shared rate limit. Empty counts per process: local runs only
    redis_url: SecretStr = SecretStr("")
    redis_timeout_seconds: float = Field(default=2, gt=0)
    # Requests per client address per minute on /v1
    rate_limit_per_minute: int = Field(default=60, gt=0)
    # Daily request caps on /v1 (0 disables caps: local runs and tests)
    daily_requests_per_caller: int = Field(default=0, ge=0)
    daily_requests_total: int = Field(default=0, ge=0)


def load_settings() -> Settings:
    """Read settings, load secrets_env_path when given, and check local GPU requirements only for local engines."""
    settings = Settings()
    if settings.secrets_env_path is not None:
        secrets_path = (SERVICE_ROOT / settings.secrets_env_path).resolve()
        if not secrets_path.is_file():
            raise SystemExit(f"Secrets file not found: {secrets_path}")
        load_dotenv(secrets_path, override=False)
    if settings.triage_engine in LOCAL_MODEL_ENGINES:
        if not os.environ.get("HF_TOKEN"):
            raise SystemExit("HF_TOKEN is empty. Local Hugging Face engines require it.")
        if settings.device == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise SystemExit("DEVICE=cuda but PyTorch sees no GPU. Install the cu128 wheel or set DEVICE=cpu.")
    return settings
