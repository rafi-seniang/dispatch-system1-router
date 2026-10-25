"""Dispatch: configurable System One router (local Laya or hosted Jev) plus the hospital inbox preset.

Run: uvicorn app:app --port 8010
"""

import os
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, TypeVar

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engines import ENGINE_CLASSES, EngineUnavailable, JevEngine, TriageEngine
from questions import (
    EngineOption,
    RoutingAction,
    SortedItem,
    SortRequest,
    TriageDecision,
    routing_action,
)
from settings import LAB_FOLDER, Settings, load_settings

VERSION = "0.4.0"
T = TypeVar("T")
PatientMessage = Annotated[str, Field(min_length=1, max_length=2000)]


class ServiceMeta(BaseModel):
    """What the lab frontend's lobby shows about this backend (the Lab backend contract)."""

    id: str
    name: str
    summary: str
    version: str
    engine: str
    own_ui_path: str
    docs_path: str


class TriageRequest(BaseModel):
    message: PatientMessage


class TriageBatchRequest(BaseModel):
    messages: list[PatientMessage] = Field(min_length=1, max_length=100)


class TriagedMessage(BaseModel):
    action: RoutingAction
    decision: TriageDecision


class TriageResponse(TriagedMessage):
    engine: str
    latency_ms: float


class TriageBatchResponse(BaseModel):
    engine: str
    latency_ms: float
    # The thresholds the actions were computed with, so a screen draws the same gate
    review_probability_threshold: float
    escalate_urgency_score: float
    results: list[TriagedMessage]


class SortResponse(BaseModel):
    engine: str
    latency_ms: float
    confidence_gate: float
    results: list[SortedItem]


class EnginesResponse(BaseModel):
    engines: list[EngineOption]


def available_engines(engines: dict[str, TriageEngine], jev_model: str) -> list[EngineOption]:
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
            detail=f"{jev_model} through api.typesafe.ai, about half a second a request."
            if "jev" in engines
            else "Add JEV_API_KEY to the secrets file to enable Jev.",
        ),
    ]


def load_engines(settings: Settings) -> dict[str, TriageEngine]:
    """The engine TRIAGE_ENGINE names, plus Jev whenever its key is set, so a request can pick either."""
    primary = ENGINE_CLASSES[settings.triage_engine](settings)
    engines = {primary.name: primary}
    if "jev" not in engines and os.environ.get("JEV_API_KEY", "").strip():
        engines["jev"] = JevEngine(settings)
    return engines


def create_app(settings: Settings | None = None, engines: dict[str, TriageEngine] | None = None) -> FastAPI:
    resolved_settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.settings = resolved_settings
        application.state.engines = engines or load_engines(resolved_settings)
        application.state.engine = application.state.engines[resolved_settings.triage_engine]
        application.state.gpu_lock = threading.Lock()  # one model on one GPU: requests take turns
        yield

    application = FastAPI(title="Dispatch: System One router", version=VERSION, lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    application.mount("/static", StaticFiles(directory=LAB_FOLDER / "static"), name="static")

    @application.get("/v1/engines")
    def list_engines(request: Request) -> EnginesResponse:
        return EnginesResponse(
            engines=available_engines(request.app.state.engines, request.app.state.settings.jev_model)
        )

    @application.post("/v1/sort")
    def sort_items(body: SortRequest, request: Request) -> SortResponse:
        active_engine: TriageEngine | None = request.app.state.engines.get(body.engine)
        if active_engine is None:
            detail = (
                "Jev needs JEV_API_KEY in the secrets file."
                if body.engine == "jev"
                else f"{body.engine} is not loaded. Set TRIAGE_ENGINE={body.engine} to serve it."
            )
            raise HTTPException(status_code=503, detail=detail)
        criteria = {option.label: option.description for option in body.choices}
        started = time.perf_counter()
        decisions = run_engine(
            request, active_engine, lambda: active_engine.sort_batch(body.items, body.instructions, criteria)
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        sorted_items = [
            SortedItem(
                choice=decision.choice,
                probabilities=decision.probabilities,
                top_probability=decision.top_probability,
                checkpoint=decision.checkpoint,
                passed_gate=decision.top_probability >= body.confidence_gate,
            )
            for decision in decisions
        ]
        return SortResponse(
            engine=active_engine.name,
            latency_ms=elapsed_ms,
            confidence_gate=body.confidence_gate,
            results=sorted_items,
        )

    @application.post("/v1/triage")
    def triage(body: TriageRequest, request: Request) -> TriageResponse:
        triaged, latency_ms = triage_messages([body.message], request)
        return TriageResponse(engine=request.app.state.engine.name, latency_ms=latency_ms, **triaged[0].model_dump())

    @application.post("/v1/triage/batch")
    def triage_batch(body: TriageBatchRequest, request: Request) -> TriageBatchResponse:
        active_settings: Settings = request.app.state.settings
        triaged, latency_ms = triage_messages(body.messages, request)
        return TriageBatchResponse(
            engine=request.app.state.engine.name,
            latency_ms=latency_ms,
            review_probability_threshold=active_settings.review_probability_threshold,
            escalate_urgency_score=active_settings.escalate_urgency_score,
            results=triaged,
        )

    @application.get("/health")
    def health(request: Request) -> dict[str, str]:
        return {"status": "ok", "engine": request.app.state.engine.name}

    @application.get("/meta")
    def meta(request: Request) -> ServiceMeta:
        return ServiceMeta(
            id="dispatch",
            name="Dispatch",
            summary="Routes any paragraph into custom choices with a System One decision model, no text generated.",
            version=VERSION,
            engine=request.app.state.engine.name,
            own_ui_path="/",
            docs_path="/docs",
        )

    @application.get("/")
    def index() -> FileResponse:
        return FileResponse(LAB_FOLDER / "static" / "index.html")

    return application


def run_engine(request: Request, engine: TriageEngine, call: Callable[[], T]) -> T:
    """One engine call: GPU engines take turns, a hosted engine runs at once, and a hosted failure is a 503."""
    try:
        if not engine.on_gpu:
            return call()
        with request.app.state.gpu_lock:
            return call()
    except EngineUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def triage_messages(messages: list[str], request: Request) -> tuple[list[TriagedMessage], float]:
    """All messages in one engine call, with the action for each and the elapsed milliseconds."""
    engine: TriageEngine = request.app.state.engine
    settings: Settings = request.app.state.settings
    started = time.perf_counter()
    decisions = run_engine(request, engine, lambda: engine.decide_batch(messages))
    triaged = [
        TriagedMessage(
            action=routing_action(decision, settings.review_probability_threshold, settings.escalate_urgency_score),
            decision=decision,
        )
        for decision in decisions
    ]
    return triaged, round((time.perf_counter() - started) * 1000, 1)


app = create_app()
