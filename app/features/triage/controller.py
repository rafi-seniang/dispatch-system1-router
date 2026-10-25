"""Turns triage requests into service calls and adds the thresholds the screen draws the gate with."""

from app.features.triage.schemas import TriageBatchRequest, TriageBatchResponse, TriageRequest, TriageResponse
from app.features.triage.service import TriageService
from app.platform.errors import unavailable_as_503
from config.settings import Settings


class TriageController:
    def __init__(self, service: TriageService, settings: Settings) -> None:
        self._service = service
        self._settings = settings

    def triage(self, body: TriageRequest) -> TriageResponse:
        triaged, latency_ms = unavailable_as_503(lambda: self._service.triage([body.message]))
        return TriageResponse(engine=self._service.engine_name, latency_ms=latency_ms, **triaged[0].model_dump())

    def triage_batch(self, body: TriageBatchRequest) -> TriageBatchResponse:
        triaged, latency_ms = unavailable_as_503(lambda: self._service.triage(body.messages))
        return TriageBatchResponse(
            engine=self._service.engine_name,
            latency_ms=latency_ms,
            review_probability_threshold=self._settings.review_probability_threshold,
            escalate_urgency_score=self._settings.escalate_urgency_score,
            results=triaged,
        )
