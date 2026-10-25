"""The general sorter: which engines are loaded, and any paragraphs sorted into custom choices."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import get_dispatch_controller
from app.controllers.dispatch import DispatchController
from app.schemas.sort import EnginesResponse, SortRequest, SortResponse

router = APIRouter(tags=["sort"])
Controller = Annotated[DispatchController, Depends(get_dispatch_controller)]


@router.get("/engines")
def list_engines(controller: Controller) -> EnginesResponse:
    return controller.engines()


# Plain def, not async: engine calls block, so FastAPI runs them in its thread pool
@router.post("/sort")
def sort_items(body: SortRequest, controller: Controller) -> SortResponse:
    return controller.sort(body)
