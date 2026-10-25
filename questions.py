"""Questions for the hospital preset and the general System One sorter."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

DEPARTMENT_INSTRUCTIONS = "Which team of the hospital should handle this patient message?"
DEPARTMENTS = {
    "appointment": "booking, rescheduling or cancelling a doctor visit or vaccination",
    "billing": "payments, invoices, deposits, insurance claims and refunds",
    "results": "lab, radiology or other test results and reports",
    "medication": "prescriptions, refills, doses and questions about medicines",
    "emergency": "symptoms that need immediate medical care right now",
}

URGENCY_INSTRUCTIONS = "How urgently must the hospital act on this message?"
URGENCY_LEVELS = [
    "routine: can wait several days",
    "today: should be handled within the day",
    "immediate: a risk to health or life right now",
]

COMPLAINT_INSTRUCTIONS = "The sender is complaining or expressing dissatisfaction with the hospital's service."


class TriageDecision(BaseModel):
    department: str
    department_probabilities: dict[str, float]
    # Probability-weighted level, 0 (routine) to 2 (immediate)
    urgency_score: float
    urgency_probabilities: list[float]
    complaint_probability: float


class ChoiceOption(BaseModel):
    label: Annotated[str, Field(min_length=1, max_length=80)]
    description: Annotated[str, Field(default="", max_length=400)] = ""


class SortDecision(BaseModel):
    choice: str
    probabilities: dict[str, float]
    top_probability: float
    checkpoint: str


class SortedItem(SortDecision):
    passed_gate: bool


class EngineOption(BaseModel):
    id: Literal["laya", "jev"]
    name: str
    provider: str
    available: bool
    status_label: str
    detail: str


def validate_unique_choices(choices: list[ChoiceOption]) -> list[ChoiceOption]:
    seen: set[str] = set()
    for option in choices:
        if not option.label.strip():
            raise ValueError("Choice label cannot be blank.")
        if option.label in seen:
            raise ValueError(f"Duplicate choice label: {option.label}")
        seen.add(option.label)
    return choices


class SortRequest(BaseModel):
    engine: Literal["laya", "jev"] = "laya"
    instructions: Annotated[str, Field(min_length=1, max_length=1000)]
    choices: Annotated[list[ChoiceOption], Field(min_length=2, max_length=20)]
    confidence_gate: Annotated[float, Field(default=0.7, ge=0.0, le=1.0)] = 0.7
    items: Annotated[list[Annotated[str, Field(min_length=1, max_length=4000)]], Field(min_length=1, max_length=100)]

    @model_validator(mode="after")
    def _unique_choices(self) -> "SortRequest":
        validate_unique_choices(self.choices)
        return self


RoutingAction = Literal["escalate_now", "auto_route", "human_review"]


def routing_action(
    decision: TriageDecision, review_probability_threshold: float, escalate_urgency_score: float
) -> RoutingAction:
    # Either signal is enough: an emergency must not depend on the urgency score alone
    if decision.department == "emergency" or decision.urgency_score >= escalate_urgency_score:
        return "escalate_now"
    if max(decision.department_probabilities.values()) < review_probability_threshold:
        return "human_review"
    return "auto_route"
