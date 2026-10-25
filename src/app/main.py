"""App factory. `uvicorn app.main:app --app-dir src` for the default settings, create_app(...) in tests."""

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import VERSION
from app.api.routes import health, sort, triage
from app.core.config import SERVICE_ROOT, Settings, load_settings
from app.core.resources import lifespan
from app.engines.base import TriageEngine


def create_app(settings: Settings | None = None, engines: dict[str, TriageEngine] | None = None) -> FastAPI:
    """`engines` replaces the loaded models, so tests run with fakes and no GPU."""
    settings = settings or load_settings()
    app = FastAPI(title="Dispatch: System One router", version=VERSION, lifespan=lifespan(settings, engines))
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"], allow_headers=["*"]
    )
    app.mount("/static", StaticFiles(directory=SERVICE_ROOT / "static"), name="static")

    api = APIRouter(prefix="/v1")
    api.include_router(sort.router)
    api.include_router(triage.router)
    app.include_router(api)
    app.include_router(health.router)
    return app


app = create_app()
