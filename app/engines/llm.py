"""A small instruct LLM as a decision model: each question becomes a lettered multiple-choice prompt, one forward pass,
and a softmax over the answer letters only."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.engines.base import decision_from_probabilities, model_dtype, sort_decision_from_probabilities
from app.engines.decisions import SortDecision, TriageDecision
from app.engines.questions import (
    COMPLAINT_INSTRUCTIONS,
    DEPARTMENT_INSTRUCTIONS,
    DEPARTMENTS,
    URGENCY_INSTRUCTIONS,
    URGENCY_LEVELS,
)
from config.settings import Settings

ANSWER_LETTERS = "ABCDE"


class LetterProbabilityLLMEngine:
    name = "llm"
    on_gpu = True

    def __init__(self, settings: Settings) -> None:
        self._device = settings.device
        self._tokenizer = AutoTokenizer.from_pretrained(settings.llm_model_id)
        self._model = AutoModelForCausalLM.from_pretrained(settings.llm_model_id, dtype=model_dtype(settings.device))
        # transformers types the wrapped .to() as taking a model, not the device name it takes
        self._model.to(settings.device).eval()  # type: ignore[arg-type]
        self._letter_token_ids = [self._single_token_id(letter) for letter in ANSWER_LETTERS]

    def _single_token_id(self, letter: str) -> int:
        token_ids = self._tokenizer.encode(letter, add_special_tokens=False)
        if len(token_ids) != 1:
            raise SystemExit(f"Answer letter {letter!r} is {len(token_ids)} tokens in this tokenizer, it must be one")
        return int(token_ids[0])

    def _prompt(self, message: str, instructions: str, options: list[str]) -> str:
        option_lines = "\n".join(f"{letter}. {option}" for letter, option in zip(ANSWER_LETTERS, options, strict=False))
        user_text = (
            "You triage patient messages sent to a hospital.\n\n"
            f"Message:\n<message>\n{message}\n</message>\n\n"
            f"Question: {instructions}\n{option_lines}\n\n"
            "Answer with the letter of one option only."
        )
        chat = [{"role": "user", "content": user_text}]
        return str(
            self._tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        )

    @torch.inference_mode()
    def _option_probabilities(self, prompt: str, option_count: int) -> list[float]:
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        next_token_logits = self._model(**inputs).logits[0, -1]
        letter_logits = next_token_logits[self._letter_token_ids[:option_count]].float()
        return torch.softmax(letter_logits, dim=0).tolist()

    def _decide(self, message: str) -> TriageDecision:
        department_options = [f"{name}: {description}" for name, description in DEPARTMENTS.items()]
        department_probabilities = self._option_probabilities(
            self._prompt(message, DEPARTMENT_INSTRUCTIONS, department_options), len(DEPARTMENTS)
        )
        urgency_probabilities = self._option_probabilities(
            self._prompt(message, URGENCY_INSTRUCTIONS, URGENCY_LEVELS), len(URGENCY_LEVELS)
        )
        complaint_yes_no = self._option_probabilities(
            self._prompt(message, f"Is this statement true? {COMPLAINT_INSTRUCTIONS}", ["yes", "no"]), 2
        )
        return decision_from_probabilities(department_probabilities, urgency_probabilities, complaint_yes_no[0])

    def decide_batch(self, messages: list[str]) -> list[TriageDecision]:
        # One prompt at a time: left-padded batches ran 2x faster but moved probabilities by up to 0.011,
        # so an answer would depend on what else was in the batch
        return [self._decide(message) for message in messages]

    def sort_batch(self, items: list[str], instructions: str, criteria: dict[str, str]) -> list[SortDecision]:
        options = [label if not description else f"{label}: {description}" for label, description in criteria.items()]
        labels = list(criteria)
        # One prompt at a time so padding never shifts single-token probabilities
        return [
            sort_decision_from_probabilities(
                labels, self._option_probabilities(self._prompt(item, instructions, options), len(options)), "llm"
            )
            for item in items
        ]
