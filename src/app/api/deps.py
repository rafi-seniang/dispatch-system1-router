"""Dependency providers. Tests swap any of them with app.dependency_overrides."""

from typing import Annotated

from fastapi import Depends, Request

from app.controllers.dispatch import DispatchController
from app.core.config import Settings
from app.core.resources import Resources
from app.services.dispatcher import Dispatcher


def get_resources(request: Request) -> Resources:
    resources: Resources = request.app.state.resources
    return resources


def get_settings(request: Request) -> Settings:
    """The settings this app instance was built with (not a global), so tests can differ."""
    settings: Settings = request.app.state.settings
    return settings


def get_dispatcher(
    resources: Annotated[Resources, Depends(get_resources)], settings: Annotated[Settings, Depends(get_settings)]
) -> Dispatcher:
    return Dispatcher(resources, settings)


def get_dispatch_controller(
    dispatcher: Annotated[Dispatcher, Depends(get_dispatcher)], settings: Annotated[Settings, Depends(get_settings)]
) -> DispatchController:
    return DispatchController(dispatcher, settings)
