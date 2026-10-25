"""Tests for the configurable System One sorter endpoints."""

from pathlib import Path

from app import create_app
from fastapi.testclient import TestClient

from questions import DEPARTMENTS, SortDecision, TriageDecision
from settings import Settings


class FakeSortEngine:
    name = "laya"

    def decide_batch(self, messages: list[str]) -> list[TriageDecision]:
        probabilities = {name: (0.8 if name == "billing" else 0.05) for name in DEPARTMENTS}
        return [
            TriageDecision(
                department="billing",
                department_probabilities=probabilities,
                urgency_score=0.4,
                urgency_probabilities=[0.7, 0.2, 0.1],
                complaint_probability=0.2,
            )
            for _ in messages
        ]

    def sort_batch(self, items: list[str], instructions: str, criteria: dict[str, str]) -> list[SortDecision]:
        labels = list(criteria.keys())
        first_label = labels[0]
        remainder = round(0.2 / (len(labels) - 1), 4)
        probabilities = {label: (0.8 if label == first_label else remainder) for label in labels}
        return [
            SortDecision(
                choice=first_label,
                probabilities=probabilities,
                top_probability=0.8,
                checkpoint="multilingual" if "Tagihan" in item else "english",
            )
            for item in items
        ]


def make_settings() -> Settings:
    return Settings(_env_file=None, secrets_env_path=Path("../../secrets/secrets.env"), device="cpu")


def test_engines_lists_laya_and_jev_coming_soon() -> None:
    with TestClient(create_app(settings=make_settings(), engine=FakeSortEngine())) as client:
        response = client.get("/v1/engines")
    assert response.status_code == 200
    engines = {item["id"]: item for item in response.json()["engines"]}
    assert engines["laya"]["available"] is True
    assert engines["jev"]["available"] is False
    assert engines["jev"]["status_label"] == "Coming soon"


def test_sort_custom_choices_returns_probabilities_and_gate() -> None:
    with TestClient(create_app(settings=make_settings(), engine=FakeSortEngine())) as client:
        response = client.post(
            "/v1/sort",
            json={
                "engine": "laya",
                "instructions": "Which topic fits this paragraph best?",
                "choices": [
                    {"label": "A", "description": "billing and invoices"},
                    {"label": "B", "description": "technical troubleshooting"},
                    {"label": "C", "description": ""},
                ],
                "confidence_gate": 0.75,
                "items": ["We were billed twice on October 12.", "Tagihan bulan ini belum masuk."],
            },
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["engine"] == "laya"
    assert len(payload["results"]) == 2
    assert payload["results"][0]["choice"] == "A"
    assert payload["results"][0]["passed_gate"] is True
    assert payload["results"][0]["checkpoint"] == "english"
    assert payload["results"][1]["checkpoint"] == "multilingual"


def test_sort_rejects_jev_while_coming_soon() -> None:
    with TestClient(create_app(settings=make_settings(), engine=FakeSortEngine())) as client:
        response = client.post(
            "/v1/sort",
            json={
                "engine": "jev",
                "instructions": "Sort into A or B",
                "choices": [{"label": "A"}, {"label": "B"}],
                "items": ["Sample text"],
            },
        )
    assert response.status_code == 503
    assert "coming soon" in response.json()["detail"].lower()


def test_sort_rejects_duplicate_choice_labels() -> None:
    with TestClient(create_app(settings=make_settings(), engine=FakeSortEngine())) as client:
        response = client.post(
            "/v1/sort",
            json={
                "engine": "laya",
                "instructions": "Sort into A or B",
                "choices": [{"label": "A"}, {"label": "A"}],
                "items": ["Sample text"],
            },
        )
    assert response.status_code == 422
