"""Smallest useful Laya call: one state, the three question types, the raw answer printed as JSON."""

import json

from laya import Router

from settings import load_settings

settings = load_settings()
router = Router(device=settings.device)

questions = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this message?",
        "criteria": {
            "billing": "payments, invoices, refunds",
            "appointment": "booking or rescheduling visits",
            "other": "anything else",
        },
    },
    "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["routine", "today", "immediate"]},
    "complaint": {"type": "noul", "instructions": "The sender is complaining about the service."},
}

for state in [
    "I was charged twice for the same MRI. This is unacceptable, refund me today.",
    "Mau ganti jadwal kontrol bulan depan ke tanggal 20 bisa?",
]:
    result = router.predict(state, questions)
    print(f"\n>>> {state}")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
