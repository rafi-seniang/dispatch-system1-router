# Dispatch: System 1 decision router (`Jev` / `Laya` / LLM logprobs)

Tried 25 October 2026 on the RTX 5050. Jev (hosted) and three open ways to get typed decisions without generating
text, compared on the same 50 labelled messages, plus a configurable System 1 router API (`POST /v1/sort`) and web
app where each request picks Jev or Laya.

## What a decision model is

An LLM writes text one token at a time. A decision model reads a **state** (a message, a ticket, a JSON record)
and answers **typed questions** about it in one forward pass. Three question types:

| Type | Returns | Example here |
|---|---|---|
| `choice` | one of your options, a probability for each | which team: appointment, billing, results, medication, emergency |
| `score` | a probability per ordered level, and their weighted mean | urgency: routine (0), today (1), immediate (2) |
| `noul` (Bernoulli, yes or no) | the probability that a statement is true | "the sender is complaining" |

Nothing is generated, so an answer cannot be outside your options and there is nothing to parse. The
probabilities are what make it useful: code can act on confident answers and send the rest to a person.

## The landscape (checked 25 October 2026)

| Name | What | Open | Notes |
|---|---|---|---|
| **Jev** (TypeSafe AI) | the model that named the category, launched 15 Sep 2026 | no, API only, early access | $0.042 per million input tokens, output free, 70 to 500 ms. Trained with "reinforcement learning for calibrated decisions" (RLCD). Up to 255 options per choice. An independent paper (arXiv 2609.37647) found it beat Qwen3.8-27B on 27 of 37 datasets. |
| **OpenAI Decisions API** | same idea, `gpt-6-luna`, `POST /v1/decisions` | no, paid API | generally available 22 Oct 2026, $0.10 per million input tokens, up to 400 questions per request, accepts images |
| **Laya** (Convai Innovations) | open answer to Jev: ModernBERT-large (421M, English) and mmBERT-base (322M, 100+ languages) with a decision head, a router picks the checkpoint by language | Apache 2.0, `pip install laya` | about 12 ms per message here. Its own README: "a fast base to specialise, not a zero-shot decision engine". Ships a fine-tuning recipe. |
| **jevless, LocalJev, OpenJevPro, patelvishwa112/jev-system-one-rlcd** | any LLM with log-probabilities, read as a decision model: options get letters, one forward pass, softmax over the letter tokens | open code | the method used by `llm` engine here |
| **OpenJev** | DiffusionGemma 26B-A4B (diffusion LLM) fills one masked slot per question in one pass, Jev-compatible API | open | needs a large GPU, free hosted tier on Codiv |
| **djev** | Jev-compatible server that returns bit-identical probabilities for identical requests | open | Qwen3.5-9B LoRA on SGLang, one H100 |
| **sys1bench** | benchmark harness for these models (calibration, wording sensitivity, negation, option count) | Apache 2.0 | found Jev better on choice and yes/no, both weak on scores, Laya weak on negation and many options |

## The engines in this folder

| Engine | How | Size on GPU |
|---|---|---|
| `jev` | TypeSafe's API: `POST https://api.typesafe.ai/v1/systemone` with `Authorization: Bearer $JEV_API_KEY`, model `jev-latest` (answered as `jev-1.14.0`). One text per request, up to 8 requests in flight | none, hosted |
| `laya` | `laya.Router`, English text to the ModernBERT checkpoint, other text to the mmBERT one | about 3 GB (the English and multilingual checkpoints) |
| `llm` | Qwen3.5-2B (Apache 2.0). Each question becomes a lettered multiple-choice prompt, one forward pass, softmax over the letters A to E only | 3.5 GB in bf16 |
| `embedding` | EmbeddingGemma 2, cosine similarity between the message and each option's description, softmax with temperature 0.02 | 1.2 GB (text only) |

## Results (`eval_triage.py`, 25 English and 25 Indonesian messages)

Routing rule (`questions.routing_action`): escalate when the department is emergency or expected urgency is at
least 1.5, send to a person when the department probability is below 0.7, otherwise route automatically.

| | Jev EN | Jev ID | Laya EN | Laya ID | Qwen3.5-2B EN | Qwen3.5-2B ID | Embedding EN | Embedding ID |
|---|---|---|---|---|---|---|---|---|
| Department accuracy | **1.00** | **1.00** | 0.96 | 0.80 | **1.00** | 0.88 | 0.84 | 0.68 |
| Complaint accuracy | **0.96** | **0.96** | **0.96** | **0.96** | **0.96** | 0.84 | 0.68 | 0.52 |
| Urgency accuracy (rounded score) | **0.80** | **0.80** | 0.40 | 0.28 | 0.48 | 0.36 | 0.56 | 0.40 |
| Emergencies escalated | **5 of 5** | **5 of 5** | 5 of 5 | 4 of 5 | 5 of 5 | 5 of 5 | 5 of 5 | 3 of 5 |
| Routed automatically to the wrong team | **0** | **0** | 0 | 1 | 0 | 0 | 0 | 1 |
| False alarms (non-emergency escalated, of 20) | 1 | **0** | 1 | 5 | 1 | 6 | 4 | 5 |
| Milliseconds per message (3 questions) | 46 (8 in flight, about 500 one at a time) | | 12 | | 286 | | 3 | |

Jev and Laya were run together on 25 October 2026 (`python scripts/eval_triage.py jev laya`, results in
`results/triage_eval_jev_laya.csv`). The Qwen and embedding columns come from the earlier run
(`results/triage_eval_laya_llm_embedding.csv`). The Qwen weights are no longer on this laptop.

What this shows, with only 50 messages (one message is 4 points of accuracy per language):

- **Jev is the best engine on every measure here.** Every department right in both languages, including all
  four Indonesian messages Laya got wrong (the dentist cancellation, the 3-hour queue, the insurance claim, and
  the child who drank floor cleaner, which Laya routed to medication at 0.78), every emergency escalated, and
  the only usable urgency score (0.80 against 0.28 to 0.56 for the others). It sent nothing to a person because
  it was confident and right on all 50. The costs: it is a paid hosted API (about 435 input tokens per message
  for the three questions, $0.042 per million), the text leaves the building, and a single request takes about
  half a second, so a batch needs parallel requests.
- **Qwen3.5-2B read as a decision model was the safest open option.** Every emergency escalated, nothing routed to the
  wrong team, and every wrong department came with a probability under 0.7, so the confidence gate caught
  them. It is 25 times slower than Laya here, partly because the fast linear-attention kernels
  (`flash-linear-attention`, `causal-conv1d`) do not install on Windows and PyTorch fallbacks run instead.
- **Laya is fast and very good at English and at the yes/no complaint question in both languages**, and it
  passed all four negation traps ("Saya TIDAK butuh ambulans", "I'm not complaining"). On Indonesian it is
  sometimes wrong with high confidence: a dentist cancellation was "emergency" at 0.99, waiting 3 hours at a
  clinic "emergency" at 0.95, and a child who drank floor cleaner "medication" at 0.78, which the app would
  have routed automatically. A confidence gate cannot catch an error made at 0.99.
- **Urgency (the `score` question) was weak for every engine.** Laya's urgency rises with the true level
  (mean 1.22, 1.36, 1.48 for routine, today, immediate) but is squeezed into 0.6 to 1.8, so it never says
  "routine". Thresholds fitted on separate labelled data would help. sys1bench reports the same weakness.
- **The embedding baseline is the fastest and the weakest.** Its probabilities come from a temperature I chose,
  not from training, so they are not calibrated.

Before using any of these on real messages: a few hundred labelled messages from the real inbox, the routing
thresholds fitted on part of them and measured on the rest, and for Laya its fine-tuning recipe on that data.

## Files

The service follows the Lab's backend layout (`docs/adr/0001-layout.md`): routes, controllers, services, engines,
schemas and core under `src/app`.

| Path | What |
|---|---|
| `src/app/api/routes/` | `GET /v1/engines`, `POST /v1/sort` (custom prompt and choices, `engine` picks Jev or Laya per request), `POST /v1/triage`, `POST /v1/triage/batch`, `/health`, `/meta`, `/` |
| `src/app/controllers/dispatch.py` | the confidence gate on each sorted item, and a missing or failing engine as a 503 that names it |
| `src/app/services/` | `Dispatcher` (picks the engine, GPU engines take turns, times each call) and `routing_action` |
| `src/app/engines/` | `JevEngine`, `LayaEngine`, `LetterProbabilityLLMEngine`, `EmbeddingSimilarityEngine` behind `sort_batch` and `decide_batch`, and the triage questions in the System One wire format |
| `src/app/schemas/` | the sort and triage request and response models |
| `src/app/core/` | settings and the lifespan that loads the engines once |
| `scripts/eval_triage.py` | runs engines on `data/messages.jsonl`, writes `results/triage_eval_*.csv` |
| `scripts/laya_quickstart.py` | the smallest Laya call, prints the raw answer JSON |
| `static/` | "Dispatch", the inbox screen: `index.html`, `inbox.js`, `inbox.css` on `app.css`, and `inbox.json` (the shift's 50 sample messages with invented senders and times) |
| `Dockerfile`, `docker-compose.yml`, `requirements-app.txt` | the app on CPU in Docker |

## Run

```powershell
copy .env.example .env                                   # once
$env:PYTHONPATH='src'
python scripts/laya_quickstart.py
python scripts/eval_triage.py jev laya                   # any of: jev laya llm embedding
python -m uvicorn app.main:app --app-dir src --port 8010 # TRIAGE_ENGINE in .env picks the engine
```

`JEV_API_KEY` in the secrets file (read, never copied) turns Jev on next to the local engine, so `POST /v1/sort`
can send `"engine": "jev"` or `"engine": "laya"`. Without the key, Jev is listed as "Needs key" and a Jev request
gets a 503 that says so. A Jev error (bad key, rate limit, timeout) is a 503 that names Jev, never a quiet switch
to Laya. `JEV_MODEL`, `JEV_TIMEOUT_SECONDS` and `JEV_MAX_CONCURRENT_REQUESTS` are settings.

Open http://localhost:8010. The screen loads the shift's 50 messages, triages them in one call, and lists what
needs a person: "Escalate now" first, then "Needs a person", each in arrival order. Opening a message shows the
model's three answers with the 0.70 gate drawn on the chosen team, and a control to send it to IGD or assign a
team. "Test a message" triages new text and adds it to the inbox. `/#message=i23` links to one message.
Assignments live in the page only: there is no case system behind it.

On this laptop GPU the whole shift triages in 0.6 to 0.8 s and one message in 30 to 100 ms with Laya. The first
request after about 20 seconds idle takes about 1 second while the GPU clocks back up.

Docker (CPU, Laya engine):

```powershell
docker compose up -d --build     # first start downloads the two Laya checkpoints into a named volume
```

The secrets file is mounted read-only for `HF_TOKEN`. The image runs as a non-root user. Measured in Docker
Desktop on this laptop's CPU: 300 to 850 ms per message, 29 s for the shift's 50 messages on first load (the
screen shows its loading state meanwhile), 3.5 GB of memory, image 2.0 GB. Laya's `predict_batch` takes
`batch_size` and `sort_by_length`, which its README measured at 1.4 to 1.8 times faster on CPU (not tried here).

## Checks

`ruff check . && ruff format --check . && mypy && pytest -q` (10 tests with fake engines, no GPU or network).
