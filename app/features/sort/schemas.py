"""The general sorter: any paragraphs, custom choices, a probability per choice and a confidence gate."""

import base64
import binascii
from collections import Counter
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from app.engines.decisions import SortDecision

EngineId = Literal["laya", "jev", "gemini"]


class ChoiceOption(BaseModel):
    label: Annotated[str, Field(min_length=1, max_length=80)]
    description: Annotated[str, Field(default="", max_length=400)] = ""


def _validate_unique_choices(choices: list[ChoiceOption]) -> None:
    labels = [option.label for option in choices]
    if any(not label.strip() for label in labels):
        raise ValueError("Choice label cannot be blank.")
    duplicates = [label for label, count in Counter(labels).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate choice label: {', '.join(duplicates)}")


class SortRequest(BaseModel):
    engine: EngineId = "laya"
    instructions: Annotated[str, Field(min_length=1, max_length=2000)]
    choices: Annotated[list[ChoiceOption], Field(min_length=2, max_length=20)]
    confidence_gate: Annotated[float, Field(default=0.7, ge=0.0, le=1.0)] = 0.7
    items: Annotated[list[Annotated[str, Field(min_length=1, max_length=4000)]], Field(min_length=1, max_length=100)]

    @model_validator(mode="after")
    def _unique_choices(self) -> "SortRequest":
        _validate_unique_choices(self.choices)
        return self


# JPEG and PNG are what both Jev and Gemini Decision read
ImageMime = Literal["image/jpeg", "image/png"]


def _starts_as(data: bytes, mime_type: ImageMime) -> bool:
    if mime_type == "image/jpeg":
        return data.startswith(b"\xff\xd8\xff")
    return data.startswith(b"\x89PNG\r\n\x1a\n")


class ImageItemInput(BaseModel):
    name: Annotated[str, Field(default="image", max_length=120)] = "image"
    mime_type: ImageMime = "image/jpeg"
    data_base64: Annotated[str, Field(min_length=16, max_length=6_000_000)]

    @model_validator(mode="after")
    def _is_the_named_image(self) -> "ImageItemInput":
        """A wrong file is the caller's mistake (422), not Gemini failing (503)."""
        try:
            data = base64.b64decode(self.data_base64, validate=True)
        except binascii.Error as error:
            raise ValueError(f"{self.name} is not base64.") from error
        if not _starts_as(data, self.mime_type):
            raise ValueError(f"{self.name} is not a {self.mime_type} picture.")
        return self


class SortImagesRequest(BaseModel):
    engine: EngineId = "jev"
    instructions: Annotated[str, Field(min_length=1, max_length=2000)]
    choices: Annotated[list[ChoiceOption], Field(min_length=2, max_length=10)]
    confidence_gate: Annotated[float, Field(default=0.7, ge=0.0, le=1.0)] = 0.7
    images: Annotated[list[ImageItemInput], Field(min_length=1, max_length=10)]

    @model_validator(mode="after")
    def _unique_choices(self) -> "SortImagesRequest":
        _validate_unique_choices(self.choices)
        return self


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
    default_engine: EngineId = "jev"
    engines: list[EngineOption]
