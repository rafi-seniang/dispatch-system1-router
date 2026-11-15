"""Gemini Decision 1: Google's hosted System One model, through the Gemini Decisions API (v1beta)."""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
from google import genai
from google.genai import types
from google.genai._interactions import APIError as NextGenAPIError
from google.genai.errors import APIError

from app.engines.base import EngineUnavailable, decision_from_answers, sort_decision_from_answer
from app.engines.decisions import SortDecision, TriageDecision
from app.engines.questions import TRIAGE_QUESTIONS, choice_question
from config.settings import Settings


class GeminiDecisionEngine:
    """Same question structure as Jev, one request per text, sent in parallel."""

    name = "gemini"
    on_gpu = False

    def __init__(self, settings: Settings, http_client: httpx.Client | None = None) -> None:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("GEMINI_API_KEY is empty in the secrets file or environment. The gemini engine needs it.")
        self._model = settings.gemini_decision_model
        self._max_concurrent_requests = settings.gemini_max_concurrent_requests
        retry = types.HttpRetryOptions(
            attempts=1 if http_client is not None else 2,
            initial_delay=0.5,
            max_delay=2.0,
        )
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                api_version="v1beta",
                timeout=int(settings.gemini_timeout_seconds * 1000),
                retry_options=retry,
                httpx_client=http_client,
            ),
        )

    def _ask(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.decisions.create(model=self._model, input=state, questions=questions)
        except NextGenAPIError as error:
            if error.status_code is not None:
                detail = (error.message or "")[:300]
                raise EngineUnavailable(f"Gemini answered {error.status_code}: {detail}") from error
            raise EngineUnavailable(f"Gemini did not answer: {type(error).__name__}") from error
        except APIError as error:
            detail = (error.message or "")[:300]
            raise EngineUnavailable(f"Gemini answered {error.code}: {detail}") from error
        except httpx.HTTPError as error:
            raise EngineUnavailable(f"Gemini did not answer: {type(error).__name__}") from error
        if not response.answers:
            raise EngineUnavailable("Gemini returned no answers.")
        return {qid: answer.model_dump() for qid, answer in response.answers.items()}

    def _ask_all(self, states: list[str], questions: dict[str, Any]) -> list[dict[str, Any]]:
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
