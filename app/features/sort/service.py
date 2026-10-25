"""The general sorter: which engines can sort, and any paragraphs sorted into custom choices by the one named."""

from app.engines.decisions import SortDecision
from app.engines.runner import EngineRunner
from app.features.sort.schemas import EngineId, EngineOption
from config.settings import Settings


class SortService:
    def __init__(self, runner: EngineRunner, settings: Settings) -> None:
        self._runner = runner
        self._settings = settings

    def engine_options(self) -> list[EngineOption]:
        engines = self._runner.engines
        jev_detail = (
            f"{self._settings.jev_model} through api.typesafe.ai, about half a second a request."
            if "jev" in engines
            else "Add JEV_API_KEY to the secrets file to enable Jev."
        )
        return [
            EngineOption(
                id="laya",
                name="Laya",
                provider="Convai Innovations (Apache 2.0, local GPU)",
                available="laya" in engines,
                status_label="Active" if "laya" in engines else "Not loaded",
                detail="ModernBERT-large for English and mmBERT-base for 100+ languages.",
            ),
            EngineOption(
                id="jev",
                name="Jev",
                provider="TypeSafe AI (hosted System One API)",
                available="jev" in engines,
                status_label="Active" if "jev" in engines else "Needs key",
                detail=jev_detail,
            ),
        ]

    def sort(
        self, engine_id: EngineId, items: list[str], instructions: str, criteria: dict[str, str]
    ) -> tuple[str, list[SortDecision], float]:
        engine = self._runner.engine(engine_id)
        decisions, latency_ms = self._runner.run(engine, lambda: engine.sort_batch(items, instructions, criteria))
        return engine.name, decisions, latency_ms
