"""Settings for the decision-model experiments, read from .env next to this file."""

import os
from pathlib import Path
from typing import Literal

import torch
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

LAB_FOLDER = Path(__file__).parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=LAB_FOLDER / ".env", extra="ignore")

    # Personal secrets file that holds HF_TOKEN (never printed)
    secrets_env_path: Path
    # Engine the triage app serves
    triage_engine: Literal["laya", "jev", "llm", "embedding"] = "laya"
    # Jev, TypeSafe's hosted System One model. JEV_API_KEY comes from the secrets file
    jev_base_url: str = "https://api.typesafe.ai"
    jev_model: str = "jev-latest"
    jev_timeout_seconds: float = 30
    # Requests in flight at once when a batch goes to Jev (one text per request)
    jev_max_concurrent_requests: int = 8
    # cuda or cpu
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


def load_settings() -> Settings:
    """Read settings and put HF_TOKEN into the environment for huggingface_hub."""
    settings = Settings()  # type: ignore[call-arg]
    secrets_path = (LAB_FOLDER / settings.secrets_env_path).resolve()
    if not secrets_path.is_file():
        raise SystemExit(f"Secrets file not found: {secrets_path}")
    load_dotenv(secrets_path, override=False)
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit(f"HF_TOKEN is empty in {secrets_path}")
    if settings.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("DEVICE=cuda but PyTorch sees no GPU. Install the cu128 wheel or set DEVICE=cpu.")
    return settings


def model_dtype(device: str) -> torch.dtype:
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32
