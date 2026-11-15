"""Liveness with the serving engine, /meta for the lab frontend, and the inbox screen at /."""

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from app import VERSION
from app.features.health.schemas import ServiceMeta
from app.platform.dependencies import get_resources
from app.platform.resources import Resources
from config.settings import SERVICE_ROOT

router = APIRouter(tags=["health"])
ResourcesDep = Annotated[Resources, Depends(get_resources)]


@router.get("/health")
def health(resources: ResourcesDep) -> dict[str, str]:
    return {"status": "ok", "engine": resources.runner.triage_engine.name}


@router.get("/meta")
def meta(resources: ResourcesDep) -> ServiceMeta:
    return ServiceMeta(
        id="dispatch",
        name="Dispatch",
        summary="Makes very fast, accurate decisions from messy text or images into custom prompt-defined choices.",
        version=VERSION,
        engine=resources.runner.triage_engine.name,
        own_ui_path="/",
        docs_path="/docs",
    )


@router.get("/")
def index() -> FileResponse:
    return FileResponse(SERVICE_ROOT / "static" / "index.html")
