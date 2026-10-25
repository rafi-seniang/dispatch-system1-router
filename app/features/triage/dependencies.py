"""How the triage feature is built per request: the shared engine runner, then service, then controller."""

from typing import Annotated

from fastapi import Depends

from app.features.triage.controller import TriageController
from app.features.triage.service import TriageService
from app.platform.dependencies import get_resources, get_settings
from app.platform.resources import Resources
from config.settings import Settings


def get_triage_service(
    resources: Annotated[Resources, Depends(get_resources)], settings: Annotated[Settings, Depends(get_settings)]
) -> TriageService:
    return TriageService(resources.runner, settings)


def get_triage_controller(
    service: Annotated[TriageService, Depends(get_triage_service)], settings: Annotated[Settings, Depends(get_settings)]
) -> TriageController:
    return TriageController(service, settings)
