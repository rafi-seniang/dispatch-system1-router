# Dispatch: a fast decision router

Sorts texts or pictures into choices you write in plain language, with a probability for each choice. It uses decision
models instead of a chat model. Part of the AI Lab (rafilab.com): it keeps the Lab contract (`/health`, `/meta`,
OpenAPI) and has a screen in the lab frontend.

## What a decision model is

An LLM writes text one token at a time. A decision model reads a **state** (a message, a ticket, a picture) and answers
**typed questions** about it in one pass, without generating anything:

| Type | Returns | Example |
|---|---|---|
| `choice` | one of your options, a probability for each | which team: billing, results, emergency |
| `score` | a probability per ordered level, and their weighted mean | urgency: routine, today, immediate |
| `noul` (yes or no) | the probability that a statement is true | "the sender is complaining" |

An answer cannot fall outside your options and there is nothing to parse. The probabilities let code act on confident
answers and send the rest to a human (the confidence gate on each item).

## Engines

| Engine | How |
|---|---|
| `jev` (default) | TypeSafe's hosted decision model (`jev-1.14.0`). One text or picture per request, up to 8 in flight. A picture goes as an image asset (PNG or JPEG). The connection is kept open for 2 minutes between requests |
| `gemini` | Google's Decisions API (`gemini-decision-1`), up to three keys, the next on a 429 |
| `laya` | open local model (`laya.Router`), for comparison and offline use |
| `llm`, `embedding` | baselines: an LLM read through letter log-probabilities, and embedding similarity |

An engine without its key is listed as unavailable, and a request to it returns a 503 that says so. An API error
returns a 503 naming that backend, never a quiet fallback. `JEV_API_KEY` and `GEMINI_API_KEY` come from
`SECRETS_ENV_PATH` or the environment.

## API

| Endpoint | What |
|---|---|
| `GET /v1/engines` | the engines and whether each is available |
| `POST /v1/sort` | 1 to 100 texts into your choices, with the engine picked per request and the confidence gate |
| `POST /v1/sort/images` | 1 to 10 JPEG or PNG pictures, refused with 422 when a file is not the named type |
| `POST /v1/triage`, `/v1/triage/batch` | the hospital triage preset (department, complaint, urgency) |
| `GET /health`, `GET /meta`, `GET /docs` | the Lab contract |

A caller is counted by address (the service takes no API key) and gets `RATE_LIMIT_PER_MINUTE` requests a minute and
optional daily caps (`DAILY_REQUESTS_PER_CALLER`, `DAILY_REQUESTS_TOTAL`), counted in Redis, then a 429 with
`Retry-After`.

## Run

```powershell
copy .env.example .env
docker compose up -d redis
$env:PYTHONPATH='.'
python main.py                                           # TRIAGE_ENGINE in .env picks the default engine
python scripts/eval_triage.py jev gemini                 # the 50-message evaluation, any of: jev gemini laya llm embedding
```

`docker compose up -d --build` builds the hosted slim image (Jev and Gemini only). A CPU Laya image needs
`--build-arg INCLUDE_LOCAL_MODELS=true`. The service also serves a small hospital inbox screen at `/` (the shift's 50
sample messages), separate from the lab frontend's Dispatch screen.

## Layout

Feature-first (`docs/adr/0002-feature-first-layout.md`): one folder per feature, shared pieces in `app/platform` and
`app/engines`.

```
main.py          entry point
config/          settings
app/
  bootstrap.py   the app factory
  platform/      resources, dependencies, errors, rate limit, Redis
  engines/       Jev, Gemini, Laya, the baselines, the triage questions, and the runner (picks the engine, times each call)
  features/
    sort/        /v1/engines, /v1/sort, /v1/sort/images
    triage/      /v1/triage and /v1/triage/batch
    health/      /health, /meta and the inbox screen
scripts/         eval_triage.py, laya_quickstart.py
static/          the hospital inbox screen
docs/            adr/ (decisions), experiments/ (measurements)
```

## Checks

`ruff check . && ruff format --check . && mypy && pytest -q` (20 tests with fake engines and mocked transports, no
GPU or network).

## Measured

- **50 triage messages** (25 English, 25 Indonesian): Jev and Gemini Decision 1 both route every message to the
  right department and escalate all 10 emergencies with no wrong automatic route. Jev rates urgency better (0.84 and
  0.76 against 0.76 and 0.60). The full comparison with Laya, an LLM and an embedding baseline is in
  `docs/experiments/engines-compared.md`.
- **Pictures:** six photographs with six custom choices, Jev 6 of 6 in 594 ms for the batch and Gemini 6 of 6 in
  1,713 ms. Six pictures show the wiring works, not a rate.
- **Speed and cost:** a call to Jev takes about 0.2 to 0.5 s from the Jakarta server, mostly the trip to TypeSafe,
  and a batch of 10 takes about as long as one. Jev costs $0.042 per million input tokens.

## Known limits

- Fifty messages is a small set: one message is 4 points of accuracy per language. Before real use it needs a few
  hundred labelled messages from the real inbox, with thresholds fitted on part and measured on the rest.
- Every hosted call sends the text (or picture) to TypeSafe or Google.
- Urgency (the `score` question) was weak for every engine.
