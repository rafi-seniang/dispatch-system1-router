"""Laya: Convai Innovations' open System One decision model (ModernBERT or mmBERT encoder plus a decision head)."""

from app.core.config import Settings
from app.engines.base import decision_from_answers, sort_decision_from_answer
from app.engines.questions import TRIAGE_QUESTIONS, choice_question
from app.schemas.sort import SortDecision
from app.schemas.triage import TriageDecision


class LayaEngine:
    name = "laya"
    on_gpu = True

    def __init__(self, settings: Settings) -> None:
        from laya import Router  # imported here so the other engines run without the laya package

        self._router = Router(device=settings.device)
        self._router.preload(["english", "multilingual"])  # the two automatic routing picks from

    def decide_batch(self, messages: list[str]) -> list[TriageDecision]:
        results = self._router.predict_batch(
            [{"state": message, "questions": TRIAGE_QUESTIONS} for message in messages]
        )
        return [decision_from_answers(result["answers"]) for result in results]

    def sort_batch(self, items: list[str], instructions: str, criteria: dict[str, str]) -> list[SortDecision]:
        questions = choice_question(instructions, criteria)
        results = self._router.predict_batch([{"state": item, "questions": questions} for item in items])
        return [
            sort_decision_from_answer(result["answers"]["choice"], str(result.get("routing", {}).get("model", "laya")))
            for result in results
        ]
