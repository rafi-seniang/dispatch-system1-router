"""Turns sort requests into service calls, applies the confidence gate, and maps engine failures to 503."""

from app.engines.decisions import ImageInputItem, SortDecision
from app.features.sort.schemas import EnginesResponse, SortedItem, SortImagesRequest, SortRequest, SortResponse
from app.features.sort.service import SortService
from app.platform.errors import unavailable_as_503


def _build_sort_response(
    engine_name: str, decisions: list[SortDecision], latency_ms: float, confidence_gate: float
) -> SortResponse:
    results = [
        SortedItem(**decision.model_dump(), passed_gate=decision.top_probability >= confidence_gate)
        for decision in decisions
    ]
    return SortResponse(engine=engine_name, latency_ms=latency_ms, confidence_gate=confidence_gate, results=results)


class SortController:
    def __init__(self, service: SortService) -> None:
        self._service = service

    def engines(self) -> EnginesResponse:
        return EnginesResponse(default_engine=self._service.default_engine(), engines=self._service.engine_options())

    def sort(self, body: SortRequest) -> SortResponse:
        criteria = {option.label: option.description for option in body.choices}
        engine_name, decisions, latency_ms = unavailable_as_503(
            lambda: self._service.sort(body.engine, body.items, body.instructions, criteria)
        )
        return _build_sort_response(engine_name, decisions, latency_ms, body.confidence_gate)

    def sort_images(self, body: SortImagesRequest) -> SortResponse:
        criteria = {option.label: option.description for option in body.choices}
        image_items = [ImageInputItem(data_base64=img.data_base64, mime_type=img.mime_type) for img in body.images]
        engine_name, decisions, latency_ms = unavailable_as_503(
            lambda: self._service.sort_images(body.engine, image_items, body.instructions, criteria)
        )
        return _build_sort_response(engine_name, decisions, latency_ms, body.confidence_gate)
