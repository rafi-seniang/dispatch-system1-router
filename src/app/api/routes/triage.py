"""The hospital inbox preset: one patient message, or a whole shift of them."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import get_dispatch_controller
from app.controllers.dispatch import DispatchController
from app.schemas.triage import TriageBatchRequest, TriageBatchResponse, TriageRequest, TriageResponse

router = APIRouter(tags=["triage"])
Controller = Annotated[DispatchController, Depends(get_dispatch_controller)]


@router.post("/triage")
def triage(body: TriageRequest, controller: Controller) -> TriageResponse:
    return controller.triage(body)


@router.post("/triage/batch")
def triage_batch(body: TriageBatchRequest, controller: Controller) -> TriageBatchResponse:
    return controller.triage_batch(body)
