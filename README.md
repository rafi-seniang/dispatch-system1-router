# Dispatch: System 1 decision router (`Jev` / `Gemini Decision 1` / `Laya` / LLM logprobs)

Tried 25 October 2026 on the RTX 5050, with Google's `gemini-decision-1` evaluated alongside Jev on 15 November 2026.
Hosted engines (Jev and Gemini Decision 1) and three open local ways to get typed decisions without generating text,
compared on the same 50 labelled messages, plus a configurable System 1 router API (`POST /v1/sort`) and web app where
each request picks `jev`, `gemini`, or `laya`.

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
| `jev` | TypeSafe's API: `POST https://api.typesafe.ai/v1/systemone` with `Authorization: Bearer $JEV_API_KEY`, model `jev-latest` (answered as `jev-1.14.0`). One text or one picture per request, up to 8 requests in flight. A picture goes as an asset (PNG or JPEG) that replaces a random placeholder in the state, with the `TypeSafe-Danger-Zone-Images: true` header | none, hosted |
| `gemini` | Google's Decisions API (`v1beta`): `POST /v1beta/decisions` via `google-genai` with `GEMINI_API_KEY`, model `gemini-decision-1`. One text or one picture per request, up to 8 requests in flight. Up to three keys, the next on a 429 | none, hosted |
| `laya` | `laya.Router`, English text to the ModernBERT checkpoint, other text to the mmBERT one | about 3 GB (the English and multilingual checkpoints) |
| `llm` | Qwen3.5-2B (Apache 2.0). Each question becomes a lettered multiple-choice prompt, one forward pass, softmax over the letters A to E only | 3.5 GB in bf16 |
| `embedding` | EmbeddingGemma 2, cosine similarity between the message and each option's description, softmax with temperature 0.02 | 1.2 GB (text only) |

## Results (`eval_triage.py`, 25 English and 25 Indonesian messages)

Routing rule (`questions.routing_action`): escalate when the department is emergency or expected urgency is at
least 1.5, send to a person when the department probability is below 0.7, otherwise route automatically.

| | Jev EN | Jev ID | Gemini EN | Gemini ID | Laya EN | Laya ID | Qwen3.5-2B EN | Qwen3.5-2B ID | Embedding EN | Embedding ID |
|---|---|---|---|---|---|---|---|---|---|---|
| Department accuracy | **1.00** | **1.00** | **1.00** | **1.00** | 0.96 | 0.80 | **1.00** | 0.88 | 0.84 | 0.68 |
| Complaint accuracy | **0.96** | **0.96** | **0.96** | 0.92 | **0.96** | **0.96** | **0.96** | 0.84 | 0.68 | 0.52 |
| Urgency accuracy (rounded score) | **0.84** | **0.76** | 0.76 | 0.60 | 0.40 | 0.28 | 0.48 | 0.36 | 0.56 | 0.40 |
| Emergencies escalated | **5 of 5** | **5 of 5** | **5 of 5** | **5 of 5** | 5 of 5 | 4 of 5 | 5 of 5 | 5 of 5 | 5 of 5 | 3 of 5 |
| Routed automatically to the wrong team | **0** | **0** | **0** | **0** | 0 | 1 | 0 | 0 | 0 | 1 |
| False alarms (non-emergency escalated, of 20) | **0** | **0** | **0** | **0** | 1 | 5 | 1 | 6 | 4 | 5 |
| Milliseconds per message (3 questions) | 35 (8 in flight) | | 76 (8 in flight) | | 12 | | 286 | | 3 | |

Gemini Decision 1 and Jev were run together on 15 November 2026 (`python scripts/eval_triage.py gemini jev`, results in
`results/triage_eval_gemini_jev.csv`). The Laya column comes from `results/triage_eval_jev_laya.csv` (25 October 2026,
where Jev scored 0.80 urgency accuracy, 1 false alarm in EN, and 46 ms per message). The Qwen and embedding columns
come from the earlier run (`results/triage_eval_laya_llm_embedding.csv`). The Qwen weights are no longer on this laptop.

What this shows, with only 50 messages (one message is 4 points of accuracy per language):

- **Jev is still the best overall engine here.** It matched 100% department accuracy, raised 0 false alarms in the
  November run, had the highest urgency accuracy (0.84 EN, 0.76 ID), and ran at 35 ms per message at 8-way concurrency.
  The costs: it is a paid hosted API (about 435 input tokens per message for the three questions, $0.042 per million),
  the text leaves the building, and a single request takes about half a second, so a batch needs parallel requests.
- **Gemini Decision 1 matches Jev on department accuracy (1.00 EN, 1.00 ID) and safety.** All 5 emergencies in each
  language escalated, with 0 wrong auto-routes and 0 false alarms, at 76 ms per message with 8 requests in flight. It
  is weaker than Jev on urgency calibration (0.76 EN, 0.60 ID, tending to score routine messages higher near 0.6 to
  0.9) and missed the double-QRIS billing complaint in Indonesian (`i03`). It exposes a free-tier option under Google
  AI Studio alongside Jev.
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

## Layout

Feature-first (`docs/adr/0002-feature-first-layout.md`): each feature is one folder, and what both features share is
in `app/platform` and `app/engines`.

```
main.py                     entry point
config/settings.py          every environment value, with a comment
app/
  bootstrap.py              the app factory: CORS, the inbox screen's static files, /v1 behind the rate limit
  platform/                 resources (engines and Redis, built once), dependencies, errors (503 and 429),
                            rate_limit, redis
  engines/                  Jev, Gemini Decision 1, Laya, the LLM letter-probability and embedding baselines behind
                            sort_batch and decide_batch, the triage questions in the System One wire format, and the
                            runner (finds the engine a request names, GPU engines take turns, times each call)
  features/
    sort/                   GET /v1/engines, POST /v1/sort (texts) and POST /v1/sort/images (1 to 10 JPEG or PNG
                            pictures, refused with 422 when a file is not the named type): custom choices, the
                            engine picked per request, the confidence gate on each item
    triage/                 POST /v1/triage and /v1/triage/batch: the hospital preset and routing_action
    health/                 /health, /meta and the inbox screen at /
scripts/                    eval_triage.py, laya_quickstart.py (run from the repository root with PYTHONPATH=.)
static/                     the inbox screen: index.html, inbox.js, inbox.css on app.css, inbox.json (the shift's 50
                            sample messages with invented senders and times)
Dockerfile, docker-compose.yml, requirements-hosted.txt, requirements-app.txt   hosted slim image by default, or CPU Laya with INCLUDE_LOCAL_MODELS=true
```

Each client address gets `RATE_LIMIT_PER_MINUTE` requests a minute on `/v1`, counted in Redis, then a 429 with
`Retry-After`. The GPU serves one model call at a time, so one client cannot queue everyone else out.

## Run

```powershell
copy .env.example .env                                   # once
docker compose up -d redis                               # the shared rate limit
$env:PYTHONPATH='.'
python scripts/laya_quickstart.py
python scripts/eval_triage.py jev laya                   # any of: jev laya llm embedding
python main.py                                           # TRIAGE_ENGINE in .env picks the engine
```

`JEV_API_KEY` and `GEMINI_API_KEY` (via `SECRETS_ENV_PATH` or the environment) turn Jev and Gemini Decision 1 on next to
any local engine, so `POST /v1/sort` can send `"engine": "jev"`, `"engine": "gemini"`, or `"engine": "laya"`. Without
a key, an engine is listed as unavailable ("Needs JEV_API_KEY" / "Needs GEMINI_API_KEY") and requests to it return a
503 that says so. An API error (bad key, rate limit, timeout) returns a 503 naming that backend, never a quiet
fallback. `DAILY_REQUESTS_PER_CALLER` and `DAILY_REQUESTS_TOTAL` enforce optional UTC daily caps via Redis or memory.
A caller is counted by address: the service takes no API key, so a sent `X-API-Key` does not start a fresh count.

Pictures, checked on 16 November 2026 with six photographs from Lens's archive (surfing, a tiger, Mars, the Burj
Khalifa, sushi, a helicopter) and six custom choices: Jev 6 of 6 right in 594 ms for the batch, Gemini Decision 1 6 of
6 in 1,713 ms. Both put Mars at 0.70, at the gate. Six pictures show the wiring works, not a rate.

Open http://localhost:8010. The screen loads the shift's 50 messages, triages them in one call, and lists what
needs a person: "Escalate now" first, then "Needs a person", each in arrival order. Opening a message shows the
model's three answers with the 0.70 gate drawn on the chosen team, and a control to send it to IGD or assign a
team. "Test a message" triages new text and adds it to the inbox. `/#message=i23` links to one message.
Assignments live in the page only: there is no case system behind it.

On this laptop GPU the whole shift triages in 0.6 to 0.8 s and one message in 30 to 100 ms with Laya. The first
request after about 20 seconds idle takes about 1 second while the GPU clocks back up.

Docker (hosted Jev/Gemini slim image by default, or CPU Laya with `--build-arg INCLUDE_LOCAL_MODELS=true`):

```powershell
docker compose up -d --build
```

For local CPU Laya in Docker, set `TRIAGE_ENGINE=laya` and build with `INCLUDE_LOCAL_MODELS=true` so the two Laya
checkpoints download into the named volume. The image runs as a non-root user. Measured earlier on CPU Laya in Docker
Desktop: 300 to 850 ms per message, 29 s for 50 messages on first load, 3.5 GB of memory, image 2.0 GB. The hosted
slim image skips PyTorch and Hugging Face packages.

## Checks

`ruff check . && ruff format --check . && mypy && pytest -q` (16 tests with fake engines and mocked transports, no GPU or network).
