"""What an engine answers: a triage decision for the hospital preset, or a sort decision for custom choices."""

from pydantic import BaseModel


class TriageDecision(BaseModel):
    department: str
    department_probabilities: dict[str, float]
    # Probability-weighted level, 0 (routine) to 2 (immediate)
    urgency_score: float
    urgency_probabilities: list[float]
    complaint_probability: float


class SortDecision(BaseModel):
    choice: str
    probabilities: dict[str, float]
    top_probability: float
    checkpoint: str
