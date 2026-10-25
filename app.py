"""Dispatch: configurable System One router (Laya active, Jev coming soon) plus the hospital inbox preset.

Run: uvicorn app:app --port 8010
"""

import os
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engines import ENGINE_CLASSES, TriageEngine
from questions import (
    EngineOption,
    RoutingAction,
    SortedItem,
    SortRequest,
    TriageDecision,
    routing_action,
)
from settings import LAB_FOLDER, Settings, load_settings

VERSION = "0.3.0"
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


def available_engines() -> list[EngineOption]:
    has_jev_key = bool(os.environ.get("JEV_API_KEY", "").strip())
    return [
        EngineOption(
            id="laya",
            name="Laya",
            provider="Convai Innovations (Apache 2.0, local GPU)",
            available=True,
            status_label="Active",
            detail="ModernBERT-large for English and mmBERT-base for 100+ languages.",
        ),
        EngineOption(
            id="jev",
            name="Jev",
            provider="TypeSafe AI (System One API)",
            available=False,
            status_label="Coming soon",
            detail="JEV_API_KEY is set, client integration coming soon."
            if has_jev_key
            else "Coming soon. Add JEV_API_KEY in secrets.env when ready.",
        ),
    ]


def create_app(settings: Settings | None = None, engine: TriageEngine | None = None) -> FastAPI:
    resolved_settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.settings = resolved_settings
        application.state.engine = engine or ENGINE_CLASSES[resolved_settings.triage_engine](resolved_settings)
        application.state.engine_lock = threading.Lock()  # one model on one GPU: requests take turns
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
    def list_engines() -> EnginesResponse:
        return EnginesResponse(engines=available_engines())

    @application.post("/v1/sort")
    def sort_items(body: SortRequest, request: Request) -> SortResponse:
        if body.engine == "jev":
            raise HTTPException(
                status_code=503,
                detail="Jev is coming soon. Select Laya for now until JEV_API_KEY is configured.",
            )
        active_engine: TriageEngine = request.app.state.engine
        criteria = {option.label: option.description for option in body.choices}
        started = time.perf_counter()
        with request.app.state.engine_lock:
            decisions = active_engine.sort_batch(body.items, body.instructions, criteria)
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


def triage_messages(messages: list[str], request: Request) -> tuple[list[TriagedMessage], float]:
    """All messages in one engine call, with the action for each and the elapsed milliseconds."""
    engine: TriageEngine = request.app.state.engine
    settings: Settings = request.app.state.settings
    started = time.perf_counter()
    with request.app.state.engine_lock:
        decisions = engine.decide_batch(messages)
    triaged = [
        TriagedMessage(
            action=routing_action(decision, settings.review_probability_threshold, settings.escalate_urgency_score),
            decision=decision,
        )
        for decision in decisions
    ]
    return triaged, round((time.perf_counter() - started) * 1000, 1)


app = create_app()
