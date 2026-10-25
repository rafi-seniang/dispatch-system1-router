"""EmbeddingGemma 2 as a zero-shot classifier: the nearest option description, with the classification prompt on both
sides."""

import torch
from sentence_transformers import SentenceTransformer

from app.engines.base import decision_from_probabilities, model_dtype, sort_decision_from_probabilities
from app.engines.decisions import SortDecision, TriageDecision
from app.engines.questions import DEPARTMENTS, URGENCY_LEVELS
from config.settings import Settings

CLASSIFICATION_PROMPT = "task: classification | query: "
COMPLAINT_OPTIONS = [
    "A complaint, the sender is unhappy with the hospital's service.",
    "A neutral question or request, no complaint.",
]


class EmbeddingSimilarityEngine:
    name = "embedding"
    on_gpu = True

    def __init__(self, settings: Settings) -> None:
        self._temperature = settings.embedding_temperature
        self._model = SentenceTransformer(
            settings.embedding_model_id,
            device=settings.device,
            model_kwargs={"dtype": model_dtype(settings.device)},
            config_kwargs={"vision_config": None, "audio_config": None},
        )
        self._department_embeddings = self._encode(
            [f"{name}: {description}" for name, description in DEPARTMENTS.items()]
        )
        self._urgency_embeddings = self._encode(URGENCY_LEVELS)
        self._complaint_embeddings = self._encode(COMPLAINT_OPTIONS)

    def _encode(self, texts: list[str]) -> torch.Tensor:
        return self._model.encode(
            texts, prompt=CLASSIFICATION_PROMPT, convert_to_tensor=True, normalize_embeddings=True
        ).float()

    def _probabilities(self, message_embeddings: torch.Tensor, option_embeddings: torch.Tensor) -> torch.Tensor:
        return torch.softmax(message_embeddings @ option_embeddings.T / self._temperature, dim=1)

    def decide_batch(self, messages: list[str]) -> list[TriageDecision]:
        message_embeddings = self._encode(messages)
        department_probabilities = self._probabilities(message_embeddings, self._department_embeddings).tolist()
        urgency_probabilities = self._probabilities(message_embeddings, self._urgency_embeddings).tolist()
        complaint_probabilities = self._probabilities(message_embeddings, self._complaint_embeddings)[:, 0].tolist()
        return [
            decision_from_probabilities(department, urgency, complaint)
            for department, urgency, complaint in zip(
                department_probabilities, urgency_probabilities, complaint_probabilities, strict=True
            )
        ]

    def sort_batch(self, items: list[str], instructions: str, criteria: dict[str, str]) -> list[SortDecision]:
        labels = list(criteria)
        option_texts = [
            f"{instructions} {label}" if not description else f"{instructions} {label}: {description}"
            for label, description in criteria.items()
        ]
        rows = self._probabilities(self._encode(items), self._encode(option_texts)).tolist()
        return [sort_decision_from_probabilities(labels, row, "embedding") for row in rows]
