"""Turns sort and triage requests into dispatcher calls, applies the confidence gate, maps engine failures to 503."""

from collections.abc import Callable
from typing import TypeVar

from fastapi import HTTPException

from app import VERSION
from app.core.config import Settings
from app.engines.base import EngineUnavailable
from app.schemas.meta import ServiceMeta
from app.schemas.sort import EnginesResponse, SortedItem, SortRequest, SortResponse
from app.schemas.triage import TriageBatchRequest, TriageBatchResponse, TriageRequest, TriageResponse
from app.services.dispatcher import Dispatcher, EngineNotLoaded

T = TypeVar("T")


def unavailable_as_503(call: Callable[[], T]) -> T:
    """A missing or failing engine is a 503 that names it, never a quiet fallback to another engine."""
    try:
        return call()
    except (EngineNotLoaded, EngineUnavailable) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


class DispatchController:
    def __init__(self, dispatcher: Dispatcher, settings: Settings) -> None:
        self._dispatcher = dispatcher
        self._settings = settings

    def engines(self) -> EnginesResponse:
        return EnginesResponse(engines=self._dispatcher.engine_options())

    def sort(self, body: SortRequest) -> SortResponse:
        criteria = {option.label: option.description for option in body.choices}
        engine_name, decisions, latency_ms = unavailable_as_503(
            lambda: self._dispatcher.sort(body.engine, body.items, body.instructions, criteria)
        )
        results = [
            SortedItem(**decision.model_dump(), passed_gate=decision.top_probability >= body.confidence_gate)
            for decision in decisions
        ]
        return SortResponse(
            engine=engine_name, latency_ms=latency_ms, confidence_gate=body.confidence_gate, results=results
        )

    def triage(self, body: TriageRequest) -> TriageResponse:
        triaged, latency_ms = unavailable_as_503(lambda: self._dispatcher.triage([body.message]))
        return TriageResponse(
            engine=self._dispatcher.triage_engine_name, latency_ms=latency_ms, **triaged[0].model_dump()
        )

    def triage_batch(self, body: TriageBatchRequest) -> TriageBatchResponse:
        triaged, latency_ms = unavailable_as_503(lambda: self._dispatcher.triage(body.messages))
        return TriageBatchResponse(
            engine=self._dispatcher.triage_engine_name,
            latency_ms=latency_ms,
            review_probability_threshold=self._settings.review_probability_threshold,
            escalate_urgency_score=self._settings.escalate_urgency_score,
            results=triaged,
        )

    def health(self) -> dict[str, str]:
        return {"status": "ok", "engine": self._dispatcher.triage_engine_name}

    def meta(self) -> ServiceMeta:
        return ServiceMeta(
            id="dispatch",
            name="Dispatch",
            summary="Routes any paragraph into custom choices with a System One decision model, no text generated.",
            version=VERSION,
            engine=self._dispatcher.triage_engine_name,
            own_ui_path="/",
            docs_path="/docs",
        )
