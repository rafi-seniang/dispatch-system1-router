"""The hospital inbox preset: one patient message, or a whole shift of them."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.features.triage.controller import TriageController
from app.features.triage.dependencies import get_triage_controller
from app.features.triage.schemas import TriageBatchRequest, TriageBatchResponse, TriageRequest, TriageResponse

router = APIRouter(tags=["triage"])
Controller = Annotated[TriageController, Depends(get_triage_controller)]


@router.post("/triage")
def triage(body: TriageRequest, controller: Controller) -> TriageResponse:
    return controller.triage(body)


@router.post("/triage/batch")
def triage_batch(body: TriageBatchRequest, controller: Controller) -> TriageBatchResponse:
    return controller.triage_batch(body)
