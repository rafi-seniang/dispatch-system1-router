"""The general sorter: any paragraphs, custom choices, a probability per choice and a confidence gate."""

from collections import Counter
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

EngineId = Literal["laya", "jev"]


class ChoiceOption(BaseModel):
    label: Annotated[str, Field(min_length=1, max_length=80)]
    description: Annotated[str, Field(default="", max_length=400)] = ""


class SortRequest(BaseModel):
    engine: EngineId = "laya"
    instructions: Annotated[str, Field(min_length=1, max_length=1000)]
    choices: Annotated[list[ChoiceOption], Field(min_length=2, max_length=20)]
    confidence_gate: Annotated[float, Field(default=0.7, ge=0.0, le=1.0)] = 0.7
    items: Annotated[list[Annotated[str, Field(min_length=1, max_length=4000)]], Field(min_length=1, max_length=100)]

    @model_validator(mode="after")
    def _unique_choices(self) -> "SortRequest":
        labels = [option.label for option in self.choices]
        if any(not label.strip() for label in labels):
            raise ValueError("Choice label cannot be blank.")
        duplicates = [label for label, count in Counter(labels).items() if count > 1]
        if duplicates:
            raise ValueError(f"Duplicate choice label: {', '.join(duplicates)}")
        return self


class SortDecision(BaseModel):
    choice: str
    probabilities: dict[str, float]
    top_probability: float
    checkpoint: str


class SortedItem(SortDecision):
    passed_gate: bool


class SortResponse(BaseModel):
    engine: str
    latency_ms: float
    confidence_gate: float
    results: list[SortedItem]


class EngineOption(BaseModel):
    id: EngineId
    name: str
    provider: str
    available: bool
    status_label: str
    detail: str


class EnginesResponse(BaseModel):
    engines: list[EngineOption]
