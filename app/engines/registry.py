"""The engines a setting can name. Heavy local engines are imported only when built."""

from collections.abc import Callable

from app.engines.base import TriageEngine
from app.engines.gemini import GeminiDecisionEngine
from app.engines.jev import JevEngine
from config.settings import Settings


def _build_laya(settings: Settings) -> TriageEngine:
    from app.engines.laya import LayaEngine

    return LayaEngine(settings)


def _build_llm(settings: Settings) -> TriageEngine:
    from app.engines.llm import LetterProbabilityLLMEngine

    return LetterProbabilityLLMEngine(settings)


def _build_embedding(settings: Settings) -> TriageEngine:
    from app.engines.embedding import EmbeddingSimilarityEngine

    return EmbeddingSimilarityEngine(settings)


ENGINE_CLASSES: dict[str, Callable[[Settings], TriageEngine]] = {
    "jev": JevEngine,
    "gemini": GeminiDecisionEngine,
    "laya": _build_laya,
    "llm": _build_llm,
    "embedding": _build_embedding,
}
