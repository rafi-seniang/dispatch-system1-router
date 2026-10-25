"""The general sorter: which engines are loaded, and any paragraphs sorted into custom choices."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.features.sort.controller import SortController
from app.features.sort.dependencies import get_sort_controller
from app.features.sort.schemas import EnginesResponse, SortRequest, SortResponse

router = APIRouter(tags=["sort"])
Controller = Annotated[SortController, Depends(get_sort_controller)]


@router.get("/engines")
def list_engines(controller: Controller) -> EnginesResponse:
    return controller.engines()


# Plain def, not async: engine calls block, so FastAPI runs them in its thread pool
@router.post("/sort")
def sort_items(body: SortRequest, controller: Controller) -> SortResponse:
    return controller.sort(body)
