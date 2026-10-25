"""The hospital inbox preset: a patient message in, the department, urgency, complaint and routing action out."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from app.engines.decisions import TriageDecision

PatientMessage = Annotated[str, Field(min_length=1, max_length=2000)]
RoutingAction = Literal["escalate_now", "auto_route", "human_review"]


class TriageRequest(BaseModel):
    message: PatientMessage


class TriageBatchRequest(BaseModel):
    messages: list[PatientMessage] = Field(min_length=1, max_length=100)


class TriagedMessage(BaseModel):
    action: RoutingAction
    decision: TriageDecision


class TriageResponse(TriagedMessage):
    engine: str
    latency_ms: float


class TriageBatchResponse(BaseModel):
    engine: str
    latency_ms: float
    # The thresholds the actions were computed with, so a screen draws the same gate
    review_probability_threshold: float
    escalate_urgency_score: float
    results: list[TriagedMessage]
