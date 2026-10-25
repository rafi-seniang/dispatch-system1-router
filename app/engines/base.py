"""What every engine provides, and the conversions from model answers to decisions that engines share."""

from typing import Any, Protocol

import torch

from app.engines.decisions import SortDecision, TriageDecision
from app.engines.questions import DEPARTMENTS


class EngineUnavailable(RuntimeError):
    """A hosted engine did not answer. The app turns this into a 503 that names it."""


class TriageEngine(Protocol):
    name: str
    # A local model on the one GPU serves one request at a time. A hosted API takes them in parallel
    on_gpu: bool

    def decide_batch(self, messages: list[str]) -> list[TriageDecision]: ...

    def sort_batch(self, items: list[str], instructions: str, criteria: dict[str, str]) -> list[SortDecision]: ...


def model_dtype(device: str) -> torch.dtype:
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


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


def decision_from_answers(answers: dict[str, Any]) -> TriageDecision:
    """Laya's and Jev's answers to TRIAGE_QUESTIONS."""
    department = answers["department"]
    urgency = answers["urgency"]
    return TriageDecision(
        department=department["choice"],
        department_probabilities=department["probabilities"],
        urgency_score=urgency["score"],
        urgency_probabilities=list(urgency["probabilities"].values()),
        complaint_probability=answers["complaint"]["noul"],
    )


def sort_decision_from_answer(answer: dict[str, Any], checkpoint: str) -> SortDecision:
    """Laya's and Jev's answer to one choice question."""
    probabilities = {label: float(probability) for label, probability in answer["probabilities"].items()}
    choice = str(answer["choice"])
    return SortDecision(
        choice=choice, probabilities=probabilities, top_probability=probabilities[choice], checkpoint=checkpoint
    )
