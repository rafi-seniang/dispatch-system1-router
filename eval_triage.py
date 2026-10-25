"""Run each engine on data/messages.jsonl and compare accuracy, routing outcomes and speed.

Usage: python eval_triage.py [engine ...]   (default: laya llm embedding)
"""

import gc
import sys
import time

import pandas as pd
import torch

from engines import ENGINE_CLASSES
from questions import routing_action
from settings import LAB_FOLDER, Settings, load_settings


def run_engine(engine_name: str, messages: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    load_started = time.perf_counter()
    engine = ENGINE_CLASSES[engine_name](settings)
    load_seconds = time.perf_counter() - load_started

    engine.decide_batch(messages["message"].head(2).tolist())  # warm-up, so timing excludes CUDA start-up
    decide_started = time.perf_counter()
    decisions = engine.decide_batch(messages["message"].tolist())
    milliseconds_per_message = (time.perf_counter() - decide_started) * 1000 / len(messages)
    print(
        f"{engine_name}: loaded in {load_seconds:.1f}s, "
        f"{milliseconds_per_message:.1f} ms per message (3 questions, batch of {len(messages)})"
    )

    del engine
    gc.collect()
    if settings.device == "cuda":
        torch.cuda.empty_cache()

    return messages.assign(
        engine=engine_name,
        predicted_department=[decision.department for decision in decisions],
        department_probability=[max(decision.department_probabilities.values()) for decision in decisions],
        urgency_score=[decision.urgency_score for decision in decisions],
        complaint_probability=[decision.complaint_probability for decision in decisions],
        action=[
            routing_action(decision, settings.review_probability_threshold, settings.escalate_urgency_score)
            for decision in decisions
        ],
        milliseconds_per_message=milliseconds_per_message,
    )


def summarise(per_message: pd.DataFrame) -> pd.DataFrame:
    scored = per_message.assign(
        department_correct=per_message["predicted_department"] == per_message["department"],
        urgency_correct=per_message["urgency_score"].round() == per_message["urgency"],
        urgency_absolute_error=(per_message["urgency_score"] - per_message["urgency"]).abs(),
        complaint_correct=(per_message["complaint_probability"] >= 0.5) == per_message["is_complaint"],
        is_emergency=per_message["department"] == "emergency",
        is_escalated=per_message["action"] == "escalate_now",
        is_auto_routed=per_message["action"] == "auto_route",
        is_human_review=per_message["action"] == "human_review",
    )
    scored["is_auto_routed_wrong"] = scored["is_auto_routed"] & ~scored["department_correct"]
    grouped = scored.groupby(["engine", "language"])
    summary = grouped.agg(
        department_accuracy=("department_correct", "mean"),
        urgency_accuracy=("urgency_correct", "mean"),
        urgency_mae=("urgency_absolute_error", "mean"),
        complaint_accuracy=("complaint_correct", "mean"),
        auto_routed=("is_auto_routed", "sum"),
        auto_routed_wrong=("is_auto_routed_wrong", "sum"),
        sent_to_person=("is_human_review", "sum"),
        ms_per_message=("milliseconds_per_message", "first"),
    )
    summary["emergencies_escalated"] = (
        scored[scored["is_emergency"]].groupby(["engine", "language"])["is_escalated"].sum()
    )
    summary["emergencies_total"] = scored[scored["is_emergency"]].groupby(["engine", "language"]).size()
    summary["false_alarms"] = scored[~scored["is_emergency"]].groupby(["engine", "language"])["is_escalated"].sum()
    return summary.round(3)


def main() -> None:
    settings = load_settings()
    engine_names = sys.argv[1:] or list(ENGINE_CLASSES)
    messages = pd.read_json(LAB_FOLDER / "data" / "messages.jsonl", lines=True)
    print(f"{len(messages)} labelled messages {messages['language'].value_counts().to_dict()}\n")

    per_message = pd.concat([run_engine(name, messages, settings) for name in engine_names], ignore_index=True)
    print(
        f"\nescalate_now: department is emergency or urgency >= {settings.escalate_urgency_score}. "
        f"human_review: department probability < {settings.review_probability_threshold}.\n"
    )
    print(summarise(per_message).T.to_string())

    wrong_department = per_message[per_message["predicted_department"] != per_message["department"]]
    print("\nWrong department, with the probability the engine gave its answer and the action taken:")
    columns = [
        "engine",
        "message_id",
        "message",
        "department",
        "predicted_department",
        "department_probability",
        "action",
    ]
    print(wrong_department[columns].round(2).to_string(index=False))

    output_path = LAB_FOLDER / "results" / f"triage_eval_{'_'.join(engine_names)}.csv"
    output_path.parent.mkdir(exist_ok=True)
    per_message.to_csv(output_path, index=False)
    print(f"\nPer-message results written to {output_path}")


if __name__ == "__main__":
    main()
