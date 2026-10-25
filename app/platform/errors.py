"""The two errors every feature can answer with: an engine that is missing or failing (503), and too many requests."""

from collections.abc import Callable
from typing import TypeVar

from fastapi import HTTPException

from app.engines.base import EngineUnavailable
from app.engines.runner import EngineNotLoaded

T = TypeVar("T")


class RateLimited(HTTPException):
    def __init__(self, detail: str, retry_after_seconds: int) -> None:
        super().__init__(status_code=429, detail=detail, headers={"Retry-After": str(retry_after_seconds)})


def unavailable_as_503(call: Callable[[], T]) -> T:
    """A missing or failing engine is a 503 that names it, never a quiet fallback to another engine."""
    try:
        return call()
    except (EngineNotLoaded, EngineUnavailable) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
