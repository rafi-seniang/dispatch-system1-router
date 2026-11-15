"""Jev: TypeSafe's hosted System One model, through its API (POST /v1/systemone)."""

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from app.engines.base import EngineUnavailable, decision_from_answers, sort_decision_from_answer
from app.engines.decisions import ImageInputItem, SortDecision, TriageDecision
from app.engines.questions import TRIAGE_QUESTIONS, choice_question
from config.settings import Settings

# TypeSafe's opt-in header, required on every request that carries an image
IMAGE_HEADERS = {"TypeSafe-Danger-Zone-Images": "true"}


def image_state(image: ImageInputItem) -> dict[str, Any]:
    """The picture as Jev's state. Jev swaps a placeholder for its asset anywhere in the text, so it is random."""
    placeholder = f"image-{uuid.uuid4().hex}"
    asset = {"type": "image", "format": image.mime_type.removeprefix("image/"), "data_base64": image.data_base64}
    return {"state": placeholder, "assets": {placeholder: asset}}


class JevEngine:
    """The same questions as Laya, one request per text, sent in parallel."""

    name = "jev"
    on_gpu = False

    def __init__(self, settings: Settings) -> None:
        api_key = os.environ.get("JEV_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("JEV_API_KEY is empty in the secrets file. The jev engine needs it.")
        self._model = settings.jev_model
        self._max_concurrent_requests = settings.jev_max_concurrent_requests
        self._http = httpx.Client(
            base_url=settings.jev_base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=settings.jev_timeout_seconds,
        )

    def _ask(self, state: str | dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        """A text is the state itself. A picture comes as image_state's state and assets."""
        content = {"state": state} if isinstance(state, str) else state
        headers = IMAGE_HEADERS if "assets" in content else None
        body = {"model": self._model, **content, "questions": questions}
        try:
            response = self._http.post("/v1/systemone", json=body, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            detail = error.response.text[:300]
            raise EngineUnavailable(f"Jev answered {error.response.status_code}: {detail}") from error
        except httpx.HTTPError as error:
            raise EngineUnavailable(f"Jev did not answer: {type(error).__name__}") from error
        result: dict[str, Any] = response.json()
        return result

    def _ask_all(self, states: list[str] | list[dict[str, Any]], questions: dict[str, Any]) -> list[dict[str, Any]]:
        # Network-bound, so a few requests in flight at once. pool.map keeps the input order
        with ThreadPoolExecutor(max_workers=self._max_concurrent_requests) as pool:
            return list(pool.map(lambda state: self._ask(state, questions), states))

    def decide_batch(self, messages: list[str]) -> list[TriageDecision]:
        return [decision_from_answers(result["answers"]) for result in self._ask_all(messages, TRIAGE_QUESTIONS)]

    def sort_batch(self, items: list[str], instructions: str, criteria: dict[str, str]) -> list[SortDecision]:
        return [
            sort_decision_from_answer(result["answers"]["choice"], str(result["model"]))
            for result in self._ask_all(items, choice_question(instructions, criteria))
        ]

    def sort_images_batch(
        self, images: list[ImageInputItem], instructions: str, criteria: dict[str, str]
    ) -> list[SortDecision]:
        states = [image_state(image) for image in images]
        return [
            sort_decision_from_answer(result["answers"]["choice"], str(result["model"]))
            for result in self._ask_all(states, choice_question(instructions, criteria))
        ]
