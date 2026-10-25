"""The hospital preset's three questions, and the same questions in the System One wire format Laya and Jev take."""

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

TRIAGE_QUESTIONS = {
    "department": {"type": "choice", "instructions": DEPARTMENT_INSTRUCTIONS, "criteria": DEPARTMENTS},
    "urgency": {"type": "score", "instructions": URGENCY_INSTRUCTIONS, "criteria": URGENCY_LEVELS},
    "complaint": {"type": "noul", "instructions": COMPLAINT_INSTRUCTIONS},
}


def choice_question(instructions: str, criteria: dict[str, str]) -> dict[str, dict[str, object]]:
    """One custom choice question, as the general sorter sends it."""
    return {"choice": {"type": "choice", "instructions": instructions, "criteria": criteria}}
