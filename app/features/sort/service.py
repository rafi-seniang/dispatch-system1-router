"""The general sorter: which engines can sort, and any paragraphs sorted into custom choices by the one named."""

from app.engines.base import EngineUnavailable
from app.engines.decisions import ImageInputItem, SortDecision
from app.engines.runner import EngineRunner
from app.features.sort.schemas import EngineId, EngineOption
from config.settings import Settings


class SortService:
    def __init__(self, runner: EngineRunner, settings: Settings) -> None:
        self._runner = runner
        self._settings = settings

    def default_engine(self) -> EngineId:
        name = self._runner.triage_engine.name
        if name == "jev":
            return "jev"
        if name == "gemini":
            return "gemini"
        if name == "laya":
            return "laya"
        if "jev" in self._runner.engines:
            return "jev"
        return "gemini" if "gemini" in self._runner.engines else "laya"

    def engine_options(self) -> list[EngineOption]:
        engines = self._runner.engines
        jev_detail = (
            f"{self._settings.jev_model} through api.typesafe.ai, about half a second a request (text + images)."
            if "jev" in engines
            else "Add JEV_API_KEY to enable Jev."
        )
        gemini_detail = (
            f"{self._settings.gemini_decision_model} through Google Gemini API (v1beta, text + images)."
            if "gemini" in engines
            else "Add GEMINI_API_KEY to enable Gemini Decision 1."
        )
        return [
            EngineOption(
                id="jev",
                name="Jev",
                provider="TypeSafe AI (hosted System One API)",
                available="jev" in engines,
                status_label="Active" if "jev" in engines else "Needs key",
                detail=jev_detail,
            ),
            EngineOption(
                id="gemini",
                name="Gemini Decision",
                provider="Google Gemini API (free tier, inputs sent to Google)",
                available="gemini" in engines,
                status_label="Active" if "gemini" in engines else "Needs key",
                detail=gemini_detail,
            ),
            EngineOption(
                id="laya",
                name="Laya",
                provider="Convai Innovations (Apache 2.0, local GPU)",
                available="laya" in engines,
                status_label="Active" if "laya" in engines else "Not loaded",
                detail="ModernBERT-large for English and mmBERT-base for 100+ languages.",
            ),
        ]

    def sort(
        self, engine_id: EngineId, items: list[str], instructions: str, criteria: dict[str, str]
    ) -> tuple[str, list[SortDecision], float]:
        engine = self._runner.engine(engine_id)
        decisions, latency_ms = self._runner.run(engine, lambda: engine.sort_batch(items, instructions, criteria))
        return engine.name, decisions, latency_ms

    def sort_images(
        self, engine_id: EngineId, images: list[ImageInputItem], instructions: str, criteria: dict[str, str]
    ) -> tuple[str, list[SortDecision], float]:
        engine = self._runner.engine(engine_id)
        sort_images_batch = getattr(engine, "sort_images_batch", None)
        if sort_images_batch is None:
            raise EngineUnavailable(f"{engine.name} only accepts text inputs. Choose jev or gemini for images.")
        decisions, latency_ms = self._runner.run(engine, lambda: sort_images_batch(images, instructions, criteria))
        return engine.name, decisions, latency_ms
