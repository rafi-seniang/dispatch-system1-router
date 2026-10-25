"""How the sort feature is built per request: the shared engine runner, then service, then controller."""

from typing import Annotated

from fastapi import Depends

from app.features.sort.controller import SortController
from app.features.sort.service import SortService
from app.platform.dependencies import get_resources, get_settings
from app.platform.resources import Resources
from config.settings import Settings


def get_sort_service(
    resources: Annotated[Resources, Depends(get_resources)], settings: Annotated[Settings, Depends(get_settings)]
) -> SortService:
    return SortService(resources.runner, settings)


def get_sort_controller(service: Annotated[SortService, Depends(get_sort_service)]) -> SortController:
    return SortController(service)
