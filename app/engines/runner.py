"""Runs the loaded engines: finds the one a request names, lets GPU engines take turns, and times every call."""

import threading
import time
from collections.abc import Callable
from typing import TypeVar

from app.engines.base import TriageEngine

T = TypeVar("T")


class EngineNotLoaded(RuntimeError):
    """The request names an engine this service did not load."""


class EngineRunner:
    def __init__(self, engines: dict[str, TriageEngine], triage_engine: TriageEngine) -> None:
        self.engines = engines
        # The engine TRIAGE_ENGINE names, which serves the hospital preset
        self.triage_engine = triage_engine
        # One model on one GPU: GPU requests take turns
        self._gpu_lock = threading.Lock()

    def engine(self, engine_id: str) -> TriageEngine:
        engine = self.engines.get(engine_id)
        if engine is None:
            if engine_id == "jev":
                raise EngineNotLoaded("Jev needs JEV_API_KEY in the secrets file.")
            raise EngineNotLoaded(f"{engine_id} is not loaded. Set TRIAGE_ENGINE={engine_id} to serve it.")
        return engine

    def run(self, engine: TriageEngine, call: Callable[[], T]) -> tuple[T, float]:
        """One engine call and its milliseconds. GPU engines take turns, a hosted engine runs at once."""
        started = time.perf_counter()
        if engine.on_gpu:
            with self._gpu_lock:
                result = call()
        else:
            result = call()
        return result, round((time.perf_counter() - started) * 1000, 1)
