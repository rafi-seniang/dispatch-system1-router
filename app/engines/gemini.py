"""Gemini Decision 1: Google's hosted System One model, through the Gemini Decisions API (v1beta)."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
from google import genai
from google.genai import types
from google.genai._interactions import APIError as NextGenAPIError
from google.genai.errors import APIError

from app.engines.base import EngineUnavailable, decision_from_answers, sort_decision_from_answer
from app.engines.decisions import ImageInputItem, SortDecision, TriageDecision
from app.engines.questions import TRIAGE_QUESTIONS, choice_question
from config.settings import Settings


def load_gemini_api_keys() -> list[str]:
    """Collect primary + backup Gemini keys from env, deduplicated in order."""
    candidates = [
        os.environ.get("GEMINI_API_KEY", ""),
        os.environ.get("GEMINI_API_KEY_2", ""),
        os.environ.get("GEMINI_API_KEY_3", ""),
        *os.environ.get("GEMINI_API_KEYS", "").split(","),
    ]
    seen: set[str] = set()
    keys: list[str] = []
    for raw in candidates:
        key = raw.strip()
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def _is_rate_limit(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    code = getattr(error, "code", None)
    if status_code == 429 or code == 429:
        return True
    detail = f"{getattr(error, 'message', '')} {error}"
    return "429" in detail or "RESOURCE_EXHAUSTED" in detail


class GeminiDecisionEngine:
    """Same question structure as Jev, supporting text or images with 3-key 429 rotation."""

    name = "gemini"
    on_gpu = False

    def __init__(self, settings: Settings, http_client: httpx.Client | None = None) -> None:
        api_keys = load_gemini_api_keys()
        if not api_keys:
            raise SystemExit("GEMINI_API_KEY is empty in the secrets file or environment. The gemini engine needs it.")
        self._model = settings.gemini_decision_model
        self._max_concurrent_requests = settings.gemini_max_concurrent_requests
        retry = types.HttpRetryOptions(
            attempts=1 if http_client is not None else 2,
            initial_delay=0.5,
            max_delay=2.0,
        )
        timeout_ms = int(settings.gemini_timeout_seconds * 1000)
        self._clients = [
            genai.Client(
                api_key=key,
                http_options=types.HttpOptions(
                    api_version="v1beta",
                    timeout=timeout_ms,
                    retry_options=retry,
                    httpx_client=http_client,
                ),
            )
            for key in api_keys
        ]
        self._cursor_lock = threading.Lock()
        self._key_cursor = 0

    def _ask(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        total = len(self._clients)
        with self._cursor_lock:
            start = self._key_cursor % total

        for offset in range(total):
            idx = (start + offset) % total
            client = self._clients[idx]
            try:
                response = client.decisions.create(model=self._model, input=state, questions=questions)
            except NextGenAPIError as error:
                if _is_rate_limit(error) and offset + 1 < total:
                    with self._cursor_lock:
                        self._key_cursor = (idx + 1) % total
                    continue
                if error.status_code is not None:
                    detail = (error.message or "")[:300]
                    raise EngineUnavailable(f"Gemini answered {error.status_code}: {detail}") from error
                raise EngineUnavailable(f"Gemini did not answer: {type(error).__name__}") from error
            except APIError as error:
                if _is_rate_limit(error) and offset + 1 < total:
                    with self._cursor_lock:
                        self._key_cursor = (idx + 1) % total
                    continue
                detail = (error.message or "")[:300]
                raise EngineUnavailable(f"Gemini answered {error.code}: {detail}") from error
            except httpx.HTTPError as error:
                raise EngineUnavailable(f"Gemini did not answer: {type(error).__name__}") from error

            with self._cursor_lock:
                self._key_cursor = idx
            if not response.answers:
                raise EngineUnavailable("Gemini returned no answers.")
            return {qid: answer.model_dump() for qid, answer in response.answers.items()}

        raise EngineUnavailable("Gemini answered 429 across all configured API keys.")

    def _ask_all(self, states: list[Any], questions: dict[str, Any]) -> list[dict[str, Any]]:
        # Network-bound, so a few requests run in flight at once while pool.map preserves input order
        with ThreadPoolExecutor(max_workers=self._max_concurrent_requests) as pool:
            return list(pool.map(lambda state: self._ask(state, questions), states))

    def decide_batch(self, messages: list[str]) -> list[TriageDecision]:
        return [decision_from_answers(answers) for answers in self._ask_all(messages, TRIAGE_QUESTIONS)]

    def sort_batch(self, items: list[str], instructions: str, criteria: dict[str, str]) -> list[SortDecision]:
        return [
            sort_decision_from_answer(answers["choice"], self._model)
            for answers in self._ask_all(items, choice_question(instructions, criteria))
        ]

    def sort_images_batch(
        self, images: list[ImageInputItem], instructions: str, criteria: dict[str, str]
    ) -> list[SortDecision]:
        states = [{"type": "image", "data": item.data_base64, "mime_type": item.mime_type} for item in images]
        return [
            sort_decision_from_answer(answers["choice"], self._model)
            for answers in self._ask_all(states, choice_question(instructions, criteria))
        ]
