"""What the inbox does with a triaged message: escalate, route on its own, or send to a person."""

from app.schemas.triage import RoutingAction, TriageDecision


def routing_action(
    decision: TriageDecision, review_probability_threshold: float, escalate_urgency_score: float
) -> RoutingAction:
    # Either signal is enough: an emergency must not depend on the urgency score alone
    if decision.department == "emergency" or decision.urgency_score >= escalate_urgency_score:
        return "escalate_now"
    if max(decision.department_probabilities.values()) < review_probability_threshold:
        return "human_review"
    return "auto_route"
