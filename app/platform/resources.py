"""The engines and Redis, built once in the lifespan and held on app.state."""

import os
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI
from redis.asyncio import Redis

from app.engines.base import TriageEngine
from app.engines.jev import JevEngine
from app.engines.registry import ENGINE_CLASSES
from app.engines.runner import EngineRunner
from app.platform.rate_limit import InMemoryRateLimiter, RateLimiter, RedisRateLimiter
from app.platform.redis import connect_redis
from config.settings import Settings


@dataclass
class Resources:
    runner: EngineRunner
    rate_limiter: RateLimiter
    redis: Redis | None = None


def load_engines(settings: Settings) -> dict[str, TriageEngine]:
    """The engine TRIAGE_ENGINE names, plus Jev whenever its key is set, so a request can pick either."""
    primary = ENGINE_CLASSES[settings.triage_engine](settings)
    engines = {primary.name: primary}
    if "jev" not in engines and os.environ.get("JEV_API_KEY", "").strip():
        engines["jev"] = JevEngine(settings)
    return engines


def lifespan(
    settings: Settings, engines: dict[str, TriageEngine] | None = None
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """`engines` replaces the loaded models, so tests run with fakes and no GPU."""

    @asynccontextmanager
    async def run(app: FastAPI) -> AsyncIterator[None]:
        loaded = engines or load_engines(settings)  # the models load here, before the first request
        redis = await connect_redis(settings)
        rate_limiter: RateLimiter = RedisRateLimiter(redis) if redis is not None else InMemoryRateLimiter()
        app.state.resources = Resources(
            runner=EngineRunner(loaded, loaded[settings.triage_engine]), rate_limiter=rate_limiter, redis=redis
        )
        yield
        if redis is not None:
            await redis.aclose()

    return run
