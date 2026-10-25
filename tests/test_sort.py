"""Tests for the configurable System One sorter endpoints and the Jev client."""

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.engines.base import EngineUnavailable
from app.engines.jev import JevEngine
from app.engines.questions import DEPARTMENTS, TRIAGE_QUESTIONS
from app.main import create_app
from app.schemas.sort import SortDecision
from app.schemas.triage import TriageDecision


class FakeSortEngine:
    name = "laya"
    on_gpu = True

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


class FakeJevEngine:
    """Picks the last choice with certainty, or fails like an unreachable API when told to."""

    name = "jev"
    on_gpu = False

    def __init__(self, down: bool = False) -> None:
        self.down = down

    def decide_batch(self, messages: list[str]) -> list[TriageDecision]:
        raise NotImplementedError

    def sort_batch(self, items: list[str], instructions: str, criteria: dict[str, str]) -> list[SortDecision]:
        if self.down:
            raise EngineUnavailable("Jev answered 401: invalid key")
        last = list(criteria)[-1]
        probabilities = {label: (1.0 if label == last else 0.0) for label in criteria}
        return [SortDecision(choice=last, probabilities=probabilities, top_probability=1.0, checkpoint="jev-1.14.0")]


SORT_BODY = {
    "instructions": "Sort into A or B",
    "choices": [{"label": "A", "description": "billing"}, {"label": "B", "description": "technical"}],
    "items": ["Sample text"],
}


def make_settings() -> Settings:
    return Settings(_env_file=None, secrets_env_path=Path("../../secrets/secrets.env"), device="cpu")


def test_engines_lists_jev_as_needing_a_key_when_it_is_not_loaded() -> None:
    with TestClient(create_app(settings=make_settings(), engines={"laya": FakeSortEngine()})) as client:
        response = client.get("/v1/engines")
    assert response.status_code == 200
    engines = {item["id"]: item for item in response.json()["engines"]}
    assert engines["laya"]["available"] is True
    assert engines["jev"]["available"] is False
    assert engines["jev"]["status_label"] == "Needs key"


def test_engines_lists_jev_as_active_when_it_is_loaded() -> None:
    engines = {"laya": FakeSortEngine(), "jev": FakeJevEngine()}
    with TestClient(create_app(settings=make_settings(), engines=engines)) as client:
        jev = {item["id"]: item for item in client.get("/v1/engines").json()["engines"]}["jev"]
    assert jev["available"] is True
    assert "jev-latest" in jev["detail"]


def test_sort_custom_choices_returns_probabilities_and_gate() -> None:
    with TestClient(create_app(settings=make_settings(), engines={"laya": FakeSortEngine()})) as client:
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


def test_sort_with_jev_uses_the_jev_engine() -> None:
    engines = {"laya": FakeSortEngine(), "jev": FakeJevEngine()}
    with TestClient(create_app(settings=make_settings(), engines=engines)) as client:
        response = client.post("/v1/sort", json={**SORT_BODY, "engine": "jev"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["engine"] == "jev"
    assert payload["results"][0]["choice"] == "B"
    assert payload["results"][0]["checkpoint"] == "jev-1.14.0"


def test_sort_with_jev_without_a_key_is_503_that_says_so() -> None:
    with TestClient(create_app(settings=make_settings(), engines={"laya": FakeSortEngine()})) as client:
        response = client.post("/v1/sort", json={**SORT_BODY, "engine": "jev"})
    assert response.status_code == 503
    assert "JEV_API_KEY" in response.json()["detail"]


def test_a_jev_failure_is_503_that_names_jev() -> None:
    engines = {"laya": FakeSortEngine(), "jev": FakeJevEngine(down=True)}
    with TestClient(create_app(settings=make_settings(), engines=engines)) as client:
        response = client.post("/v1/sort", json={**SORT_BODY, "engine": "jev"})
    assert response.status_code == 503
    assert response.json()["detail"] == "Jev answered 401: invalid key"


def test_sort_rejects_duplicate_choice_labels() -> None:
    with TestClient(create_app(settings=make_settings(), engines={"laya": FakeSortEngine()})) as client:
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


def jev_engine_against(handler: httpx.MockTransport, monkeypatch: pytest.MonkeyPatch) -> JevEngine:
    monkeypatch.setenv("JEV_API_KEY", "apikey_test")
    engine = JevEngine(make_settings())
    engine._http = httpx.Client(
        base_url="https://api.typesafe.ai", headers={"Authorization": "Bearer apikey_test"}, transport=handler
    )
    return engine


def test_jev_engine_sends_the_wire_format_and_reads_the_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = json.loads(request.content)
        if "department" in body["questions"]:
            answers = {
                "department": {"type": "choice", "choice": "emergency",
                               "probabilities": {name: float(name == "emergency") for name in DEPARTMENTS}},
                "urgency": {"type": "score", "score": 2.0, "probabilities": {"0": 0.0, "1": 0.0, "2": 1.0}},
                "complaint": {"type": "noul", "noul": 0.03},
            }  # fmt: skip
        else:
            answers = {"choice": {"type": "choice", "choice": "B", "probabilities": {"A": 0.1, "B": 0.9}}}
        return httpx.Response(200, json={"model": "jev-1.14.0", "answers": answers})

    engine = jev_engine_against(httpx.MockTransport(answer), monkeypatch)
    triage = engine.decide_batch(["Anak saya minum cairan pembersih lantai"])[0]
    sorted_items = engine.sort_batch(["one", "two"], "Pick A or B", {"A": "first", "B": "second"})

    assert seen[0].url.path == "/v1/systemone"
    assert seen[0].headers["Authorization"] == "Bearer apikey_test"
    assert json.loads(seen[0].content)["questions"] == json.loads(json.dumps(TRIAGE_QUESTIONS))
    assert (triage.department, triage.urgency_score, triage.complaint_probability) == ("emergency", 2.0, 0.03)
    assert triage.urgency_probabilities == [0.0, 0.0, 1.0]
    assert [item.choice for item in sorted_items] == ["B", "B"]
    assert sorted_items[0].checkpoint == "jev-1.14.0"


def test_jev_engine_turns_an_error_into_engine_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"detail": {"error_type": "authentication_error"}})

    engine = jev_engine_against(httpx.MockTransport(refuse), monkeypatch)
    with pytest.raises(EngineUnavailable, match="Jev answered 403"):
        engine.sort_batch(["text"], "Pick A or B", {"A": "", "B": ""})


def test_jev_engine_refuses_to_start_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="JEV_API_KEY"):
        JevEngine(make_settings())
