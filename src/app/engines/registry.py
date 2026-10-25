"""The engines a setting can name. Each is built from the settings, once, at startup."""

from collections.abc import Callable

from app.core.config import Settings
from app.engines.base import TriageEngine
from app.engines.embedding import EmbeddingSimilarityEngine
from app.engines.jev import JevEngine
from app.engines.laya import LayaEngine
from app.engines.llm import LetterProbabilityLLMEngine

ENGINE_CLASSES: dict[str, Callable[[Settings], TriageEngine]] = {
    "jev": JevEngine,
    "laya": LayaEngine,
    "llm": LetterProbabilityLLMEngine,
    "embedding": EmbeddingSimilarityEngine,
}
