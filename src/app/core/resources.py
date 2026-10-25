"""The engines and the GPU lock, built once in the lifespan and held on app.state."""

import os
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI

from app.core.config import Settings
from app.engines.base import TriageEngine
from app.engines.jev import JevEngine
from app.engines.registry import ENGINE_CLASSES


@dataclass
class Resources:
    engines: dict[str, TriageEngine]
    # The engine TRIAGE_ENGINE names, which serves the hospital preset
    triage_engine: TriageEngine
    # One model on one GPU: GPU requests take turns
    gpu_lock: threading.Lock = field(default_factory=threading.Lock)


def load_engines(settings: Settings) -> dict[str, TriageEngine]:
    """The engine TRIAGE_ENGINE names, plus Jev whenever its key is set, so a request can pick either."""
    primary = ENGINE_CLASSES[settings.triage_engine](settings)
    engines = {primary.name: primary}
    if "jev" not in engines and os.environ.get("JEV_API_KEY", "").strip():
        engines["jev"] = JevEngine(settings)
    return engines


def lifespan(
    settings: Settings, engines: dict[str, TriageEngine] | None = None
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def run(app: FastAPI) -> AsyncIterator[None]:
        loaded = engines or load_engines(settings)  # the models load here, before the first request
        app.state.resources = Resources(engines=loaded, triage_engine=loaded[settings.triage_engine])
        yield

    return run
