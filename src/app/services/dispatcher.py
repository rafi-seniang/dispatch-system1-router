"""Runs the engines: picks the one a request names, lets GPU engines take turns, and times every call."""

import time
from collections.abc import Callable
from typing import TypeVar

from app.core.config import Settings
from app.core.resources import Resources
from app.engines.base import TriageEngine
from app.schemas.sort import EngineId, EngineOption, SortDecision
from app.schemas.triage import TriagedMessage
from app.services.routing import routing_action

T = TypeVar("T")


class EngineNotLoaded(RuntimeError):
    """The request names an engine this service did not load."""


class Dispatcher:
    def __init__(self, resources: Resources, settings: Settings) -> None:
        self._resources = resources
        self._settings = settings

    @property
    def triage_engine_name(self) -> str:
        return self._resources.triage_engine.name

    def engine_options(self) -> list[EngineOption]:
        engines = self._resources.engines
        jev_detail = (
            f"{self._settings.jev_model} through api.typesafe.ai, about half a second a request."
            if "jev" in engines
            else "Add JEV_API_KEY to the secrets file to enable Jev."
        )
        return [
            EngineOption(
                id="laya",
                name="Laya",
                provider="Convai Innovations (Apache 2.0, local GPU)",
                available="laya" in engines,
                status_label="Active" if "laya" in engines else "Not loaded",
                detail="ModernBERT-large for English and mmBERT-base for 100+ languages.",
            ),
            EngineOption(
                id="jev",
                name="Jev",
                provider="TypeSafe AI (hosted System One API)",
                available="jev" in engines,
                status_label="Active" if "jev" in engines else "Needs key",
                detail=jev_detail,
            ),
        ]

    def _engine(self, engine_id: EngineId) -> TriageEngine:
        engine = self._resources.engines.get(engine_id)
        if engine is None:
            if engine_id == "jev":
                raise EngineNotLoaded("Jev needs JEV_API_KEY in the secrets file.")
            raise EngineNotLoaded(f"{engine_id} is not loaded. Set TRIAGE_ENGINE={engine_id} to serve it.")
        return engine

    def _run(self, engine: TriageEngine, call: Callable[[], T]) -> tuple[T, float]:
        """One engine call and its milliseconds. GPU engines take turns, a hosted engine runs at once."""
        started = time.perf_counter()
        if engine.on_gpu:
            with self._resources.gpu_lock:
                result = call()
        else:
            result = call()
        return result, round((time.perf_counter() - started) * 1000, 1)

    def sort(
        self, engine_id: EngineId, items: list[str], instructions: str, criteria: dict[str, str]
    ) -> tuple[str, list[SortDecision], float]:
        engine = self._engine(engine_id)
        decisions, latency_ms = self._run(engine, lambda: engine.sort_batch(items, instructions, criteria))
        return engine.name, decisions, latency_ms

    def triage(self, messages: list[str]) -> tuple[list[TriagedMessage], float]:
        """All messages in one call of the triage engine, with the routing action for each."""
        engine = self._resources.triage_engine
        decisions, latency_ms = self._run(engine, lambda: engine.decide_batch(messages))
        settings = self._settings
        triaged = [
            TriagedMessage(
                action=routing_action(decision, settings.review_probability_threshold, settings.escalate_urgency_score),
                decision=decision,
            )
            for decision in decisions
        ]
        return triaged, latency_ms
