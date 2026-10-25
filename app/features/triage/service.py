"""The hospital inbox: every message triaged in one engine call, and what to do with each."""

from app.engines.decisions import TriageDecision
from app.engines.runner import EngineRunner
from app.features.triage.schemas import RoutingAction, TriagedMessage
from config.settings import Settings


def routing_action(
    decision: TriageDecision, review_probability_threshold: float, escalate_urgency_score: float
) -> RoutingAction:
    # Either signal is enough: an emergency must not depend on the urgency score alone
    if decision.department == "emergency" or decision.urgency_score >= escalate_urgency_score:
        return "escalate_now"
    if max(decision.department_probabilities.values()) < review_probability_threshold:
        return "human_review"
    return "auto_route"


class TriageService:
    def __init__(self, runner: EngineRunner, settings: Settings) -> None:
        self._runner = runner
        self._settings = settings

    @property
    def engine_name(self) -> str:
        return self._runner.triage_engine.name

    def triage(self, messages: list[str]) -> tuple[list[TriagedMessage], float]:
        """All messages in one call of the triage engine, with the routing action for each."""
        engine = self._runner.triage_engine
        decisions, latency_ms = self._runner.run(engine, lambda: engine.decide_batch(messages))
        settings = self._settings
        triaged = [
            TriagedMessage(
                action=routing_action(decision, settings.review_probability_threshold, settings.escalate_urgency_score),
                decision=decision,
            )
            for decision in decisions
        ]
        return triaged, latency_ms
