"""Turns sort requests into service calls, applies the confidence gate, and maps engine failures to 503."""

from app.features.sort.schemas import EnginesResponse, SortedItem, SortRequest, SortResponse
from app.features.sort.service import SortService
from app.platform.errors import unavailable_as_503


class SortController:
    def __init__(self, service: SortService) -> None:
        self._service = service

    def engines(self) -> EnginesResponse:
        return EnginesResponse(engines=self._service.engine_options())

    def sort(self, body: SortRequest) -> SortResponse:
        criteria = {option.label: option.description for option in body.choices}
        engine_name, decisions, latency_ms = unavailable_as_503(
            lambda: self._service.sort(body.engine, body.items, body.instructions, criteria)
        )
        results = [
            SortedItem(**decision.model_dump(), passed_gate=decision.top_probability >= body.confidence_gate)
            for decision in decisions
        ]
        return SortResponse(
            engine=engine_name, latency_ms=latency_ms, confidence_gate=body.confidence_gate, results=results
        )
