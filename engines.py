"""Three ways to answer the triage questions without generating text.

laya       open System One decision model (ModernBERT or mmBERT encoder plus a decision head)
llm        a small instruct LLM, reading the probability of each answer letter after one forward pass
embedding  EmbeddingGemma 2, cosine similarity between the message and each option's description
"""

from typing import Protocol

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from questions import (
    COMPLAINT_INSTRUCTIONS,
    DEPARTMENT_INSTRUCTIONS,
    DEPARTMENTS,
    URGENCY_INSTRUCTIONS,
    URGENCY_LEVELS,
    SortDecision,
    TriageDecision,
)
from settings import Settings, model_dtype


class TriageEngine(Protocol):
    name: str

    def decide_batch(self, messages: list[str]) -> list[TriageDecision]: ...

    def sort_batch(self, items: list[str], instructions: str, criteria: dict[str, str]) -> list[SortDecision]: ...


def expected_level(level_probabilities: list[float]) -> float:
    return sum(level * probability for level, probability in enumerate(level_probabilities))


def sort_decision_from_probabilities(labels: list[str], row: list[float], checkpoint: str) -> SortDecision:
    probabilities = dict(zip(labels, row, strict=True))
    choice = max(probabilities, key=probabilities.__getitem__)
    return SortDecision(
        choice=choice,
        probabilities=probabilities,
        top_probability=probabilities[choice],
        checkpoint=checkpoint,
    )


def decision_from_probabilities(
    department_probabilities: list[float], urgency_probabilities: list[float], complaint_probability: float
) -> TriageDecision:
    """Shared by the llm and embedding engines, which both produce plain probability lists."""
    probability_by_department = dict(zip(DEPARTMENTS, department_probabilities, strict=True))
    return TriageDecision(
        department=max(probability_by_department, key=probability_by_department.__getitem__),
        department_probabilities=probability_by_department,
        urgency_score=expected_level(urgency_probabilities),
        urgency_probabilities=urgency_probabilities,
        complaint_probability=complaint_probability,
    )


class LayaEngine:
    name = "laya"

    def __init__(self, settings: Settings) -> None:
        from laya import Router  # imported here so the other engines run without the laya package

        self._router = Router(device=settings.device)
        self._router.preload(["english", "multilingual"])  # the two automatic routing picks from
        self._questions = {
            "department": {"type": "choice", "instructions": DEPARTMENT_INSTRUCTIONS, "criteria": DEPARTMENTS},
            "urgency": {"type": "score", "instructions": URGENCY_INSTRUCTIONS, "criteria": URGENCY_LEVELS},
            "complaint": {"type": "noul", "instructions": COMPLAINT_INSTRUCTIONS},
        }

    def decide_batch(self, messages: list[str]) -> list[TriageDecision]:
        results = self._router.predict_batch([{"state": message, "questions": self._questions} for message in messages])
        return [self._to_decision(result["answers"]) for result in results]

    def sort_batch(self, items: list[str], instructions: str, criteria: dict[str, str]) -> list[SortDecision]:
        questions = {"choice": {"type": "choice", "instructions": instructions, "criteria": criteria}}
        results = self._router.predict_batch([{"state": item, "questions": questions} for item in items])
        return [self._to_sort_decision(result) for result in results]

    @staticmethod
    def _to_decision(answers: dict) -> TriageDecision:
        department = answers["department"]
        urgency = answers["urgency"]
        return TriageDecision(
            department=department["choice"],
            department_probabilities=department["probabilities"],
            urgency_score=urgency["score"],
            urgency_probabilities=list(urgency["probabilities"].values()),
            complaint_probability=answers["complaint"]["noul"],
        )

    @staticmethod
    def _to_sort_decision(result: dict) -> SortDecision:
        answer = result["answers"]["choice"]
        probabilities = {label: float(probability) for label, probability in answer["probabilities"].items()}
        choice = str(answer["choice"])
        checkpoint = str(result.get("routing", {}).get("model", "laya"))
        return SortDecision(
            choice=choice,
            probabilities=probabilities,
            top_probability=probabilities.get(choice, max(probabilities.values(), default=0.0)),
            checkpoint=checkpoint,
        )


class LetterProbabilityLLMEngine:
    """Each question becomes a lettered multiple-choice prompt: one forward pass, a softmax over the letters only."""

    name = "llm"

    def __init__(self, settings: Settings) -> None:
        self._device = settings.device
        self._tokenizer = AutoTokenizer.from_pretrained(settings.llm_model_id)
        self._model = AutoModelForCausalLM.from_pretrained(settings.llm_model_id, dtype=model_dtype(settings.device))
        self._model.to(settings.device).eval()
        self._letters = "ABCDE"
        self._letter_token_ids = [self._single_token_id(letter) for letter in self._letters]

    def _single_token_id(self, letter: str) -> int:
        token_ids = self._tokenizer.encode(letter, add_special_tokens=False)
        if len(token_ids) != 1:
            raise SystemExit(f"Answer letter {letter!r} is {len(token_ids)} tokens in this tokenizer, it must be one")
        return token_ids[0]

    def _prompt(self, message: str, instructions: str, options: list[str]) -> str:
        option_lines = "\n".join(f"{letter}. {option}" for letter, option in zip(self._letters, options, strict=False))
        user_text = (
            "You triage patient messages sent to a hospital.\n\n"
            f"Message:\n<message>\n{message}\n</message>\n\n"
            f"Question: {instructions}\n{option_lines}\n\n"
            "Answer with the letter of one option only."
        )
        return self._tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    @torch.inference_mode()
    def _option_probabilities(self, prompt: str, option_count: int) -> list[float]:
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        next_token_logits = self._model(**inputs).logits[0, -1]
        letter_logits = next_token_logits[self._letter_token_ids[:option_count]].float()
        return torch.softmax(letter_logits, dim=0).tolist()

    def decide_batch(self, messages: list[str]) -> list[TriageDecision]:
        department_options = [f"{name}: {description}" for name, description in DEPARTMENTS.items()]
        decisions = []
        # One prompt at a time: left-padded batches ran 2x faster but moved probabilities by up to 0.011,
        # so an answer would depend on what else was in the batch

        for message in messages:
            department_probabilities = self._option_probabilities(
                self._prompt(message, DEPARTMENT_INSTRUCTIONS, department_options), len(DEPARTMENTS)
            )
            urgency_probabilities = self._option_probabilities(
                self._prompt(message, URGENCY_INSTRUCTIONS, URGENCY_LEVELS), len(URGENCY_LEVELS)
            )
            complaint_yes_no = self._option_probabilities(
                self._prompt(message, f"Is this statement true? {COMPLAINT_INSTRUCTIONS}", ["yes", "no"]), 2
            )
            decisions.append(
                decision_from_probabilities(department_probabilities, urgency_probabilities, complaint_yes_no[0])
            )
        return decisions

    def sort_batch(self, items: list[str], instructions: str, criteria: dict[str, str]) -> list[SortDecision]:
        options = [label if not desc else f"{label}: {desc}" for label, desc in criteria.items()]
        labels = list(criteria.keys())
        # One prompt at a time so padding never shifts single-token probabilities
        decisions: list[SortDecision] = []
        for item in items:
            raw_probabilities = self._option_probabilities(self._prompt(item, instructions, options), len(options))
            probabilities = dict(zip(labels, raw_probabilities, strict=True))
            choice = max(probabilities, key=probabilities.__getitem__)
            decisions.append(
                SortDecision(
                    choice=choice,
                    probabilities=probabilities,
                    top_probability=probabilities[choice],
                    checkpoint="llm",
                )
            )
        return decisions


class EmbeddingSimilarityEngine:
    """Zero-shot classification by nearest option description, with the classification prompt on both sides."""

    name = "embedding"

    def __init__(self, settings: Settings) -> None:
        self._temperature = settings.embedding_temperature
        self._model = SentenceTransformer(
            settings.embedding_model_id,
            device=settings.device,
            model_kwargs={"dtype": model_dtype(settings.device)},
            config_kwargs={"vision_config": None, "audio_config": None},
        )
        self._prompt = "task: classification | query: "
        self._department_embeddings = self._encode(
            [f"{name}: {description}" for name, description in DEPARTMENTS.items()]
        )
        self._urgency_embeddings = self._encode(URGENCY_LEVELS)
        self._complaint_embeddings = self._encode(
            [
                "A complaint, the sender is unhappy with the hospital's service.",
                "A neutral question or request, no complaint.",
            ]
        )

    def _encode(self, texts: list[str]) -> torch.Tensor:
        return self._model.encode(texts, prompt=self._prompt, convert_to_tensor=True, normalize_embeddings=True).float()

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
        labels = list(criteria.keys())
        option_texts = [
            f"{instructions} {label}" if not desc else f"{instructions} {label}: {desc}"
            for label, desc in criteria.items()
        ]
        item_embeddings = self._encode(items)
        option_embeddings = self._encode(option_texts)
        rows = self._probabilities(item_embeddings, option_embeddings).tolist()
        return [sort_decision_from_probabilities(labels, row, "embedding") for row in rows]


ENGINE_CLASSES: dict[str, type[TriageEngine]] = {
    "laya": LayaEngine,
    "llm": LetterProbabilityLLMEngine,
    "embedding": EmbeddingSimilarityEngine,
}
