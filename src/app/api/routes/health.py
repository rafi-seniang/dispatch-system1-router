"""Liveness with the serving engine, /meta for the lab frontend, and the service's own inbox screen at /."""

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from app.api.deps import get_dispatch_controller
from app.controllers.dispatch import DispatchController
from app.core.config import SERVICE_ROOT
from app.schemas.meta import ServiceMeta

router = APIRouter(tags=["health"])
Controller = Annotated[DispatchController, Depends(get_dispatch_controller)]


@router.get("/health")
def health(controller: Controller) -> dict[str, str]:
    return controller.health()


@router.get("/meta")
def meta(controller: Controller) -> ServiceMeta:
    return controller.meta()


@router.get("/")
def index() -> FileResponse:
    return FileResponse(SERVICE_ROOT / "static" / "index.html")
