"""The app factory: settings, CORS, the inbox screen's static files, then one router per feature."""

from fastapi import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import VERSION
from app.engines.base import TriageEngine
from app.features.health import router as health
from app.features.sort import router as sort
from app.features.triage import router as triage
from app.platform.dependencies import enforce_rate_limit
from app.platform.resources import lifespan
from config.settings import SERVICE_ROOT, Settings, load_settings


def create_app(settings: Settings | None = None, engines: dict[str, TriageEngine] | None = None) -> FastAPI:
    """`engines` replaces the loaded models, so tests run with fakes and no GPU."""
    settings = settings or load_settings()
    app = FastAPI(title="Dispatch: System One router", version=VERSION, lifespan=lifespan(settings, engines))
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"], allow_headers=["*"]
    )
    app.mount("/static", StaticFiles(directory=SERVICE_ROOT / "static"), name="static")

    api = APIRouter(prefix="/v1", dependencies=[Depends(enforce_rate_limit)])
    api.include_router(sort.router)
    api.include_router(triage.router)
    app.include_router(api)
    app.include_router(health.router)
    return app
