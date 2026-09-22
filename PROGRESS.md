# Progress

What works today, how to run it, and what is known to be missing. Updated at the end of every
phase.

| Phase | Status |
|---|---|
| 1. Foundation | ✅ done |
| 2. Streaming ingestion | ✅ done |
| 3. Features | ✅ done |
| 4. Labels and training data | ✅ done |
| 5. Model | ✅ done |
| 6. Serving | ✅ done |
| 7. Orchestration and analytics | ✅ done |
| 8. GenAI assistant | ✅ done (except the model calls) |
| 9. Analyst app and dashboard | ✅ done (Superset unverified) |
| 10. Polish | ✅ done |
| + Azure OpenAI provider | ✅ done (not run against a real deployment) |

---

## Azure OpenAI provider ✅

Added after phase 10, at your request. The assistant now runs against either a local model or a
hosted Azure OpenAI deployment, chosen by two lines in `.env`.

### What is in place

- **`LLM_PROVIDER`** (`ollama` | `azure_openai`) and **`EMBEDDING_PROVIDER`**
  (`sentence_transformers` | `azure_openai`), independent of each other.
- **`AzureOpenAIGenerator`** and **`AzureOpenAIEmbedder`**, behind the `SummaryGenerator` and
  `Embedder` Protocols that already existed. Nothing in `summarise_case`, the grounding checks,
  the eval harness or the analyst app knows which provider it got.
- **`build_generator(settings)` / `build_embedder(settings)`** — the four call sites now ask for
  a provider rather than constructing one.
- **`make llm-check`** — one tiny request to each configured provider, so a wrong deployment name
  is found in two seconds rather than forty minutes into `make llm-eval`.
- **`make up-ai-hosted`** — core + pgvector without the 4 GB ollama container.
- **24 new tests** in `tests/test_providers.py`. No network and no subscription needed.

### The default is unchanged and still free

`LLM_PROVIDER=ollama` remains the default, so a clone of this repository runs with no Azure
subscription and no bill. That was a deliberate constraint, not an oversight: a portfolio project
that only works with someone else's paid credentials cannot be looked at by the person you want
looking at it.

### The trap this introduced, and what stops it

Both embedders emit 384 unit-length floats, so a table holding vectors from both is accepted by
Postgres, returns similarity scores between 0 and 1, and shows five plausible "similar cases"
that are not similar to anything. No error, no warning, no implausible value.

Each row now records the embedder that wrote it, and retrieval refuses to run against a
mismatched store. **Changing `EMBEDDING_PROVIDER` means `make load-cases` again.** Rows from
before the column existed are marked `unknown` and do not block retrieval; the schema carries an
`ALTER TABLE ... ADD COLUMN IF NOT EXISTS` so an existing store upgrades in place.

### Verified

- `make lint` — clean.
- `make test` — full fast suite green, including the 24 new provider tests.
- The Azure path exercised over **real HTTP against a local stub of the Azure REST API**: URL
  construction, the `api-key` header, `response_format: json_object`, 64-item batching, response
  reordering by `index`, the 401 and 429 paths, and a full `summarise_case` through to a valid,
  grounded summary (182 ms, `azure/gpt-4o-mini`).
- `make llm-check` against that stub: both providers OK, 384 dimensions.
- A wrong key fails fast with a 401 rather than being retried into a bill.

### Not verified

- **No call has been made to a real Azure OpenAI deployment.** The request shapes are pinned by
  tests against a stub I wrote, which means they are pinned to my reading of the API docs. Run
  `make llm-check` once with real credentials — that is the two-second version of this check.
- Costs are unmeasured. Nothing in the README claims a cost figure.

### If you set it up

1. Create an Azure OpenAI resource, deploy a chat model and an embedding model, and note the
   **deployment** names — they are chosen by you and often differ from the model names.
2. `cp .env.example .env`, fill in the endpoint, key and deployment names, set both providers.
3. `make llm-check` — confirm both answer before spending anything.
4. `make load-cases` — the vector store must be rebuilt with whichever embedder you chose.
5. `make llm-eval` — fills the four assistant rows in the README results table with the provider
   name recorded alongside them.

**Never commit the key.** `.env` is gitignored; `.env.example` ships an empty value.

---

## Phase 10 — Polish ✅

### What is in place

- **`docs/interview_notes.md`** — precision vs recall, why accuracy is useless here, leakage and
  point-in-time correctness, online vs offline features, SHAP, the RAG retrieval, how the LLM is
  kept grounded, threshold trade-offs, and ten likely interview questions answered from this code.
- **`scripts/readme_metrics.py`** (`make readme-metrics`) — the README results table is now
  *generated* from `reports/*.json`. It cannot carry a number a run did not produce.
- **`make metrics`** — runs every measurement and refreshes the table.
- Full README pass: design decisions, honest limitations, and what production would do differently.

### The honesty rule, made structural

`metrics.json` now records which dataset produced it, and the table generator **withholds**
model-quality rows unless the run used the real IEEE-CIS data — rather than printing a synthetic
PR-AUC with a footnote. The same applies to the assistant: a scripted generator or the stub
embedder means those rows read *not yet measured*, because "100.0%" beside a caveat is still read
as 100%.

Current state of the table: **2 of 13 metrics measured** — the two that are genuine measurements of
the system rather than of the data (scoring latency and feature drift). Everything else names the
command that would fill it in.

`tests/test_readme_metrics.py` tests that rule directly, including that a real dataset and a real
generator *are* published.

### CI, and the failure I had not checked

**Phase 8's CI run failed and I had not looked.** The fast-test job's marker expression excluded
`needs_spark` and `needs_ollama` but not `needs_pgvector`, which had been added that same phase, so
the job collected the case-store tests and errored on a missing psycopg driver. `make test` had the
same gap, so it would have failed on your laptop too.

Fixed in two ways: the marker lists are corrected, and `tests/conftest.py` now **skips tests whose
infrastructure is genuinely absent** rather than erroring. A laptop with no Postgres skips those
twelve tests with a readable reason; the dedicated CI job, which provides the service, still runs
them for real. Verified in both directions — 12 passed with Postgres up, 12 skipped with it down.

### Verified

- Fast suite **260 passed** (248 + 12 pgvector when the service is up), Spark **54**, ruff clean.
- `make readme-metrics` regenerates the table from the report files.
- Git remote updated to the renamed repository.

---

## Phase 9 — Analyst app and dashboard ✅

### What is in place

- **`analyst_app/app.py`** (`make app`) — the review queue: decision and why, payment detail, SHAP
  reasons, similar confirmed cases, an on-demand LLM summary, and two verdict buttons.
- **`analyst_app/reviews.py`** — analyst decisions in Postgres, and the **feedback loop**: a
  resolved case becomes a confirmed case in pgvector, marked `analyst_review` so it is never
  mistaken for a settled chargeback.
- **`analyst_app/pages/1_Metrics.py`** — the metrics dashboard: stat tiles, decision mix, fraud
  value caught vs missed, precision/recall per model version, latency percentiles, rule hit rates.
- **`analyst_app/theme.py`** — status palette for decisions, validated categorical slots for
  everything else, one y-axis, a table under every chart.
- **`warehouse/publish.py`** (`make publish`) — a read-only warehouse snapshot for dashboards and
  Parquet exports for Power BI.
- **`dashboard/superset/README.md`** + a `dashboard` Compose profile.

### How to run it

```bash
make app           # review queue + metrics at http://localhost:8501
make publish       # read-only snapshot + Power BI exports
make dashboard     # Superset (optional, unverified)
```

### Verified

Run on 2026-09-20 against the live stack, **driven in a real browser** (Chromium via Playwright),
not just health-checked:

- Both pages render with **no exceptions**; the metrics page draws **5 charts**.
- The queue showed **100 real flagged payments** with genuine SHAP reasons — e.g. payment 2987794,
  review, score 0.2607, `C4 = 3.0 (+1.056)`, `amount = 203.982 (+0.948)`,
  `p_emaildomain = yahoo.com (+0.652)` — and a truncated card token, no PII.
- Metrics page: 848 payments scored, 113 in the review queue, 26 blocked, 13,707 fraud value
  caught, 63 missed, 0.12% false positive rate.
- **The feedback loop was exercised by clicking the buttons.** Confirmed in Postgres afterwards:
  `analyst_reviews` holds `2987794 | fraud` and `2987780 | legitimate`, and `fraud_cases` holds
  `2987794 | t | analyst_review` — now retrievable by the next analyst.
- `make publish` produced a 1.6 MB read-only snapshot and 4 Parquet exports.
- Fast suite **248 passed**; Spark **54**; pgvector **12**; ruff clean.

Screenshots in `docs/screenshots/`, captured from these runs.

### Four defects found by opening the page rather than health-checking it

1. **`ModuleNotFoundError: No module named 'analyst_app'`** — Streamlit puts the script's directory
   on `sys.path`, not the repo root, and the package was missing from the project's package list.
   A 200 from `/_stcore/health` says the server started, not that the page renders.
2. **Every chart titled "undefined"** — passing `title=None` to Plotly leaves a title object with
   undefined text, which it renders literally.
3. **One day of data drew a lone dot** on a time axis spanning two milliseconds. A single period is
   now grouped bars on a category axis.
4. **Clicking a verdict appeared to do nothing** — `st.rerun()` wiped the confirmation. It is now
   stashed in session state and rendered on the next run.

A fifth, found while fixing the tests: both stores took their table name from a module global, and
the test fixture's patch-and-restore corrupted the schema string between cases. The table is now a
constructor argument, so a test points a store at its own table without touching module state.

### Not verified (needs your laptop)

- **Superset.** Its image could never be pulled here. `dashboard/superset/README.md` says so
  plainly and is the most likely thing in the repository to need a fix on first run.
- The analyst app *inside its container* — the code is the same, the image is not built here.

---

## Phase 8 — GenAI assistant ✅ (except the two model calls)

### What is in place

- **`genai/case_store.py`** — Postgres + pgvector. Only confirmed cases, cosine similarity, a
  payment never retrieves itself, and the IVFFlat index is deferred until the table is big enough
  to need it.
- **`genai/embeddings.py`** — `SentenceTransformerEmbedder` (real) and `HashingEmbedder`
  (deterministic, no download). The fallback is opt-in and never silent.
- **`genai/facts.py`** — the facts object. Everything the model is allowed to know, plus
  `unsupported_numbers()`, which is what makes grounding checkable.
- **`genai/prompts.py` / `genai/summarise.py`** — prompt, Pydantic schema, JSON parsing, grounding
  check, one repair attempt, and withholding the summary entirely if it still cannot be trusted.
- **`genai/load_cases.py`** (`make load-cases`) — embeds confirmed decisions into the store.
- **`eval/llm_eval.py`** (`make llm-eval`) — factual accuracy, schema validity, retrieval quality
  and latency over ≥50 flagged cases → `reports/llm_eval.json`.

### How to run it

```bash
make up-ai                         # ollama + pgvector
make ollama-pull                   # llama3.2:3b, once (~2 GB)
make load-cases                    # embed confirmed past cases
make llm-eval                      # -> reports/llm_eval.json
```

### Verified

Run on 2026-09-20 against **real Postgres 16 with pgvector 0.6.0**:

- Case store: **848 confirmed cases** loaded (31 fraud, 817 legitimate), retrieval returning
  correctly ranked neighbours.
- `eval/llm_eval.py` ran the full harness over **50 flagged cases** with confirmed labels —
  retrieval from the real store, facts built from real audit records, every metric computed and
  written to `reports/llm_eval.json`.
- Fast suite — **241 passed**; Spark — **54**; pgvector — **7**; ruff clean.

### Not verified — and why the numbers are not published

Hugging Face and the Ollama registry both return **403** from this environment's egress policy, so
neither model could be downloaded. That means:

- **the embeddings are the hashing stub**, not `all-MiniLM-L6-v2`;
- **the generator is scripted**, not `llama3.2:3b`.

So `reports/llm_eval.json` currently contains a *harness* measurement, not a model measurement, and
it says so in its own `note` field. **No LLM eval number goes in the README until you have run
`make llm-eval` against real Ollama.** The one number worth reporting from here is that the harness
works end to end and its metrics move: with scripted responses containing deliberate faults it
reported 84% schema validity and caught every planted ungrounded figure.

### Known issues / deliberate gaps

- The repair loop is one attempt. More and a small model tends to drift rather than converge, and
  the analyst is waiting.
- Factual accuracy is measured on the *delivered* summary, so a case that was repaired counts as
  accurate. `repairs_needed` is reported alongside it so the retry cost stays visible.
- The analyst feedback loop (an analyst's own decision becoming a new case) is phase 9.

---

## Phase 7 — Orchestration and analytics ✅

### What is in place

- **`airflow/dags/fraud_batch_daily.py`** — land decisions, Bronze → Silver (quality-gated) → Gold,
  reload chargebacks, export, `dbt build`, drift report.
- **`airflow/dags/fraud_retrain_weekly.py`** — rebuild the training set *as of the run date*,
  retrain, register, and promote to champion only on a better PR-AUC.
- **`streaming/decisions_sink.py`** — the decisions topic → Delta, so analytics can never slow
  down scoring.
- **`warehouse/export.py`** (`make export`) — Parquet snapshots of every Delta layer for DuckDB.
- **`dbt/`** — 7 models (3 staging, 4 marts) and **33 tests**, including two bespoke ones: every
  flagged decision must be attributable to a rule or a model, and no label may be used before its
  chargeback arrived.
- **`quality/drift.py`** (`make drift`) — Evidently feature and prediction drift, HTML for a human
  and JSON for Airflow.
- **`docker/airflow.Dockerfile`** + an `orchestration` Compose profile.

### How to run it

```bash
make warehouse       # export -> dbt build -> drift, in one go
make airflow         # Airflow at http://localhost:8080
make airflow-logs    # the standalone admin password is printed here
```

### Verified

Run on 2026-09-20 against the real services from the integration run above — real Kafka, real
Redis, real MLflow — and orchestrated by a **real Airflow 2.10.3 scheduler**:

- `airflow dags list-import-errors` — none.
- **`fraud_batch_daily`: all 9 tasks SUCCESS, DagRun state=success.** That is real Spark jobs, the
  quality gate, `dbt build` with its tests, and the drift report, in dependency order.
- **`fraud_retrain_weekly`: all 5 tasks SUCCESS.** Version 2 trained and registered, scored
  identically to the incumbent, and was correctly **left as a challenger** — the promotion rule
  doing its job.
- `dbt build` standalone — **PASS=40, ERROR=0** (7 models, 33 tests) on real pipeline data.
- Fast suite — **221 passed**; Spark suite — **54 passed**; ruff clean.

Real warehouse output, from the pipeline's own decisions:

| payments | reviewed | blocked | fraud blocked | fraud missed | false blocks | FPR |
|---|---|---|---|---|---|---|
| 848 | 113 | 26 | 25 | 1 | 1 | 0.12% |

848 payments from 6,807 scoring attempts — the deduplication in `stg_decisions` working, since the
load tests scored the same fixture repeatedly.

### Fixed along the way

**Evidently's `DataDriftPreset` emits two metrics** — a summary and a per-column table — and the
first version read the summary's position for both. The report said *"0 of 18 columns drifted
(66.7%)"*: internally contradictory, and exactly what a monitoring tool must never say. Now looked
up by metric name, with a test asserting the count and the share describe the same thing.

### Not verified yet (needs your laptop)

- **The Airflow container.** The DAGs and every job they call have been run for real, but
  `docker/airflow.Dockerfile` has never been built — Docker Hub base images are blocked here.
  It is the same gap as the rest of Compose.

### Known issues / deliberate gaps

- Airflow runs as `airflow standalone` with SQLite: right for a laptop, explicitly not production.
- Silver and Gold are full rebuilds each run, so the daily DAG's cost grows with total data rather
  than with the day's volume.
- The warehouse is a snapshot, so dashboards are as fresh as the last export.

---

## Integration run against real services (2026-09-20)

Everything below phases 1-6 had only ever been tested against fakes. It has now been run against
**real Redis, a real Kafka broker and a real MLflow server** - installed natively rather than in
containers, because this environment's egress policy blocks Docker Hub image layers.

### What ran, end to end

| Step | Result |
|---|---|
| `producer.replay` → real Kafka | 1000 events, 958/s, across 3 partitions (227/510/263) |
| Spark Structured Streaming ← Kafka | 1000 rows → Bronze Delta, partitioned by `event_date` |
| Online features → real Redis | 44 card states, keyed by token (`fp:card:<32 hex>`) |
| Bronze → Silver | 1000 rows, **quality 12/12**, no card identifiers |
| Silver → Gold | 1000 rows |
| Chargeback labels | 1000 rows; 0 known on the day of the last payment |
| Training set, default `as_of` | **0 rows** + the explanatory error - the delayed-label logic working |
| Training set, `--as-of 2023-03-15` | 1000 rows, 3.50% fraud |
| `training.train` → real MLflow | experiment created, run logged, **version 1 registered and promoted to champion** |
| API startup | loaded `mlflow:fraud-xgboost@champion`, `pii_salt_configured: true` |
| `serving.consumer` | 300 payments scored (257 allow / 36 review / 7 block), 0 failed |
| Decisions → Kafka | 300 records on the `decisions` topic |
| Audit record | token not card, 18 features, latency, model version, reason - no PII |

### Latency, measured against real services

Real Redis, the champion model from the real registry, decisions to a real broker; one four-core
host. **Not docker compose** - no container network between the processes - so treat it as a floor.

| Concurrency | p50 | p95 | p99 | req/s |
|---|---|---|---|---|
| 1 | 12.7 ms | 20.3 ms | 21.4 ms | 71 |
| 2 | 27.0 ms | 42.8 ms | 49.4 ms | 68 |
| 4 | 62.9 ms | 94.9 ms | 125.9 ms | 62 |
| 8 | 135.3 ms | 195.9 ms | 290.9 ms | 58 |

The real Redis hop costs ~1.3 ms against the in-process fake. Written to `reports/latency.json`.

### The bug it found

**`mlflow.xgboost.load_model` does not round-trip `enable_categorical`** - a model logged with it
`True` comes back `False`. Predictions are unaffected (verified identical to 1e-6), but
`shap.TreeExplainer` reads the flag when it is constructed, so **every flagged payment reached the
analyst with an empty reasons list**. The only evidence was a warning in the API log.

Fixed by restoring the parameter before the explainer is built. Two regression tests added; both
would have caught it. The in-process tests could not - it takes a model that has actually been
through a registry.

Verified live afterwards: 7 of 7 known frauds flagged, all 7 with SHAP reasons, zero warnings.

### Still needs your laptop

- **`docker compose up` itself.** The Compose file parses and its profiles resolve, but no image
  has ever been pulled or built here, so image tags, healthchecks and the container network are
  unverified. That is now the only substantial gap.
- The latency number *through* containers, which is the one the README should eventually quote.
- Ollama and pgvector (phase 8) and Airflow (phase 7) have not been run at all.

---

## Phase 6 — Serving ✅

### What is in place

- **`rules/rules.yaml` + `serving/rules.py`** — a config-driven rules engine with five shipped
  rules (blocklist, hard amount limit, extreme velocity, new card + large amount, anonymised
  recipient). No `eval`; unknown operators fail at load time.
- **`serving/scoring.py`** — the scoring path: Redis state → shared feature definitions → rules →
  model → SHAP (flagged payments only) → audit record.
- **`serving/model.py`** — loads whatever carries the `champion` alias in the MLflow registry,
  with its thresholds, and degrades to rules-only if the registry is unreachable.
- **`serving/app.py`** — `POST /score` and a `/health` that reports which model version is serving.
  The request body is the same `PaymentEvent` that travels on Kafka, so there is one schema.
- **`serving/decisions.py`** — the audit record (transaction, model version, features used,
  rule/threshold that triggered, SHAP reasons, latency) and its sinks: Kafka, JSONL, in-memory.
- **`serving/consumer.py`** (`make consume`) — reads the payments topic and scores through the API,
  committing offsets only after a payment has been scored.
- **`scripts/load_test.py`** (`make load-test`) — sweeps concurrency levels and writes
  `reports/latency.json`.
- **`tests/demo_server.py`** (`make demo-api`) — runs the whole API with no Docker at all.

### How to run it

```bash
make up && make produce ARGS="--limit 5000"
make consume                       # score the topic through the API
make load-test                     # -> reports/latency.json

make demo-api                      # or run the API with no Docker at all
```

### Verified

Run on 2026-09-20 in the development container (4 cores):

- Fast suite — **203 passed** in 21 s.
- Spark suite — **54 passed**.
- `make lint` — clean.
- End-to-end through HTTP: 2,000 payments scored against a live uvicorn server (allow 1,932 /
  review 56 / block 12), every one with an audit record.

**Latency, measured** — in-process demo server, fakeredis, no broker, four cores.
*This is not the docker-compose number:* there is no network hop to Redis and no broker behind the
audit sink.

| Concurrency | p50 | p95 | p99 | req/s |
|---|---|---|---|---|
| 1 | 11.4 ms | 14.0 ms | 19.5 ms | 84 |
| 2 | 23.4 ms | 32.3 ms | 41.2 ms | 81 |
| 4 | 49.6 ms | 75.2 ms | 92.3 ms | 78 |
| 8 | 105.5 ms | 150.2 ms | 237.6 ms | 75 |

### Fixed along the way

The first load test returned **p50 526 ms** against a 100 ms target. Profiling the path rather than
guessing found two causes, both now fixed:

- **`build_matrix` cost 21.5 ms on a single row** — it is written for training sets, and on one row
  it constructs 57 Series to hold one value each. `build_row` does the same job in 1.0 ms, with a
  test asserting the two produce identical output.
- **XGBoost used four threads per request** — eight concurrent requests on four cores meant 32
  threads competing. The served model is now pinned to one thread.

Result: p50 526 ms → 11.4 ms unqueued, throughput 15 → 84 req/s.

Throughput is flat at ~80 req/s across every concurrency level, which says the service is CPU-bound
in a single Python process — past that, the number being measured is the queue, not the service.
The production fix is more uvicorn workers.

### Verified since

All of this has now been run against real Redis, Kafka and MLflow - see the integration run at the
top of this file, including the explainer bug it uncovered. Only `docker compose` itself remains
untested.

### Known issues / deliberate gaps

- Decisions are published to Kafka but nothing lands them in Delta yet — that consumer is phase 7.
- The audit record stores feature values but not the raw payment; Bronze already has that.
- Single uvicorn worker. Fine for a laptop demo, and the reason throughput plateaus.

---

## Phase 5 — Model ✅

### What is in place

- **`training/preprocessing.py`** — the model input schema, defined once for training and for
  serving: 57 columns in a fixed order, categorical vocabularies in code (no encoder artifact to
  ship separately), and `other` kept distinct from `missing`.
- **`training/split.py`** — time-based train/validation/test split. Never random.
- **`training/thresholds.py`** — cost-based tuning of the two thresholds. Exact search over a
  candidate grid using cumulative sums, so a 100×100 grid is instant on a million rows.
- **`training/evaluate.py`** — PR-AUC, precision/recall at the chosen operating points, false
  positive rate, and the fraud **value** caught versus missed. No accuracy anywhere.
- **`training/explain.py`** — SHAP top reasons per decision, plus global feature importance logged
  with every run.
- **`training/train.py`** (`make train`) — the whole cycle: split, fit with `scale_pos_weight`,
  tune thresholds on validation, evaluate on test, log to MLflow, register, and promote to
  champion only if it beats the incumbent's PR-AUC.

### How to run it

```bash
make labels && make dataset ARGS="--as-of 2023-04-01"
make train                      # -> reports/metrics.json, reports/thresholds.json
make train ARGS="--no-mlflow"   # without a tracking server
```

### Verified

Run on 2026-09-20. Full chain executed head to tail on synthetic data — Bronze → Silver → Gold →
labels → point-in-time training set (Delta) → `python -m training.train` → report files on disk:

```
train 5600 rows (to 2023-01-04), validation 1200 rows, test 1200 rows
thresholds: review >= 0.045, block >= 0.655 (expected cost 524 vs 12,941 doing nothing)
test:       PR-AUC 0.949 | precision 0.933, recall 0.700 | FPR 0.0017
            fraud value caught 98.5% (134 missed of 9,225)
```

**These numbers are from synthetic data and mean nothing about real performance.** They are
recorded here only as evidence that the pipeline runs and writes the files the README will quote
from. The real ones come from your laptop with the Kaggle data.

- Fast suite — **153 passed** in 19 s (includes a real XGBoost fit, SHAP, and MLflow registry
  round-trips against SQLite).
- Spark suite — **54 passed** in 96 s.
- `make lint` — clean.

### Changed along the way

- **The synthetic fixture was made harder.** The first training run scored a PR-AUC of *1.000* —
  the generator's classes were perfectly separable, which tests nothing and looks fabricated. The
  distributions now overlap heavily and one giveaway was removed (the recipient email domain was
  present for every fraud and absent for most legitimate payments). The committed fixture was
  regenerated; all downstream tests still pass.
- **MLflow API**: `log_model(..., artifact_path=...)`, not `name=` — the latter is MLflow 3.x.

### Not verified yet (needs your laptop)

- MLflow **server** — the registry is exercised against SQLite in tests, not against the `mlflow`
  container.
- Training on the real IEEE-CIS data. Every number above is synthetic.

### Known issues / deliberate gaps

- The review band assumes analysts resolve every case correctly, so it looks slightly cheaper than
  it would in production.
- No model is served yet — the API still reports `model_loaded: false`. That is phase 6.

---

## Phase 4 — Labels and training data ✅

### What is in place

- **`training/chargebacks.py`** — the delayed label simulation. Fraud is confirmed by a chargeback
  7–60 days after the payment; everything else is presumed legitimate after 60 quiet days. The
  delay is derived from the transaction ID, so loading the table a day at a time produces exactly
  the same timeline as building it in one pass.
- **`training/build_labels.py`** (`make labels`) — writes the chargeback Delta table. The only
  place in the platform that reads `isFraud`.
- **`training/dataset.py`** (`make dataset`) — the point-in-time training set builder: a payment is
  included only when it has happened *and* its label had arrived by the training date. Two
  policies for immature payments (`exclude`, `assume_legitimate`), with assumed labels flagged.
- **`tests/test_point_in_time.py`** — the leakage tests, covering both feature leakage and label
  leakage, each demonstrating the failure it prevents.

### How to run it

```bash
make labels                              # build the chargeback table
make dataset                             # point-in-time training set at "now"
make dataset ARGS="--as-of 2023-04-01"   # or as of any date
make dataset ARGS="--immature-policy assume_legitimate"
```

### Verified

Run on 2026-09-19 (Spark 3.5.3 on Java 17). Measured on the 1000-row fixture:

| Training date | Labels known | Fraud rate among known |
|---|---|---|
| day of the last payment | 0 (0%) | — |
| + 10 days | 2 (0.2%) | 1.000 |
| + 30 days | 16 (1.6%) | 1.000 |
| + 61 days | 1000 (100%) | 0.035 |

Training sets built from those: at +30 days `exclude` gives **16 rows, all fraud**;
`assume_legitimate` gives **1000 rows at 1.6% fraud with 984 assumed labels**; at +61 days,
**1000 rows at 3.50% fraud** — the true rate.

- Fast suite — **100 passed**.
- Spark suite — **54 passed**, including 10 point-in-time tests.
- `make lint` — clean.

### Known issues / deliberate gaps

- The fixture covers half a day, so every payment matures at the same moment and a +30 day training
  set is 100% fraud. On the real six-month dataset the immature window is only the tail. Worth
  remembering before reading anything into a fixture-based training set.
- The training set is a full rebuild per `as_of`. Correct; not incremental.
- Nothing is trained yet — that is phase 5.

---

## Phase 3 — Features ✅

### What is in place

- **`features/definitions.py`** — every behavioural feature, defined once. 18 features across four
  groups: stateless (amount, hour, night), velocity (10m/1h/24h counts, time since last payment),
  spending pattern (lifetime average, ratio to it) and novelty (new card, new device, new email
  domain). Pure functions over a `CardState`, so a feature can only ever see the past.
- **`features/store.py`** — the Redis online store. Keyed by **card token**, not card: a Redis dump
  contains no card identifiers. Stores state rather than features (see below), with a TTL.
- **`common/pii.py` + `common/pii_spark.py`** — HMAC-SHA256 tokenisation with a secret salt, one
  implementation used by both the API and the vectorised Spark path.
- **`streaming/features.py`** — the job the `spark` container now runs: one read of Kafka, Bronze
  written and each card's Redis state updated from the same micro-batch, repartitioned by card and
  sorted so the fold happens in order.
- **`streaming/silver.py`** — Bronze → Silver: deduplicate, drop non-positive amounts, tokenise,
  and drop the four columns the token was built from. Refuses to write if quality checks fail.
- **`features/offline.py` + `streaming/gold.py`** — the batch implementation of the same features
  as Spark window functions, and the Gold table built from it.
- **`quality/expectations.py`** — a 12-expectation Great Expectations suite. The strongest of them
  is the exact column set, which fails if a raw card identifier ever reappears in Silver.

### How to run it

```bash
make up                            # kafka, spark (bronze + redis), redis, mlflow, api
make produce ARGS="--limit 5000"   # replay payments
make bronze-peek                   # what landed
make features                      # Silver + Gold (runs the quality suite on the way)
make quality                       # quality suite on its own -> reports/quality_silver.json
```

### Verified

Run on 2026-09-19 in the development container (Spark 3.5.3 on Java 17, Delta 3.2.1, GE 0.18.22):

- `make lint` — clean.
- Fast suite — **88 passed** in 1.7 s.
- Spark suite — **44 passed** in 54 s.
- **The consistency test**: all 18 features agree between the online fold and the Spark window
  implementation, to within 1e-6, across a 400-payment sample — checked one feature at a time so a
  failure would name the culprit.
- Manual end-to-end over the 1000-row fixture: Bronze → Silver (1000 rows, 24 columns, **no
  `card1` / `addr1` / `card_key`**) → quality **12/12 expectations passed** → Gold (1000 rows,
  36 columns). Online path folded the same events into 44 card states in Redis, and the busiest
  card came back with 83 lifetime payments and an average amount of 32.12.

### Fixed along the way

- **`PYSPARK_PYTHON`**: Spark was launching Python workers with the system interpreter, so the
  first pandas UDF failed with "No module named pandas". The session builder now points Spark at
  `sys.executable`. This would have hit you too.
- **Java 21 vs Arrow**: pandas UDFs fail on Java 21 with an error a long way from its cause
  (`sun.misc.Unsafe ... not available`). Confirmed by running the same tests on both JVMs; the
  session builder now warns when it finds a JVM newer than 17.
- **Feature windows** now exclude events *after* the payment being scored, not just before the
  cutoff — protection against a late-arriving payment seeing its own future.

### Not verified yet (needs your laptop)

- Redis itself: every test runs against `fakeredis`. The client code is the real `redis` package,
  but nothing here has talked to a Redis server.
- The `spark` container running the combined Bronze + Redis job against a live broker.

### Known issues / deliberate gaps

- **Known divergence**: Redis keeps the 20 most recent distinct devices per card; the offline
  window keeps all of them. A test asserts no card in the data comes near that bound, so the
  divergence cannot creep in unnoticed.
- Dropping `card1`/`addr1` costs model performance — a deliberate trade, documented in
  `docs/design_decisions.md`.
- Silver and Gold are full rebuilds, not incremental. Correct and simple; slow once the dataset is
  large.
- No labels yet, so nothing is trained on these features. That is phase 4.

---

## Phase 2 — Streaming ingestion ✅

### What is in place

- **`common/timeline.py`** — the `TransactionDT` → calendar mapping, used by every component that
  needs a date. Pure functions, fully tested, so the producer, the streaming job and (later) the
  chargeback simulation cannot disagree about what "day 40" means.
- **`common/events.py`** — the `PaymentEvent` schema that travels on Kafka. Carries **no label**
  (a real payment message has none), splits the sparse C/D/M/V/identity blocks into maps, and
  builds the card identity proxy used as the partition key.
- **`producer/replay.py`** — replays transactions in `TransactionDT` order at a configurable
  speed-up, keyed by card so Kafka keeps each card's payments in sequence. The sink and the clock
  are injected, so ordering, keying and pacing are all tested without a broker.
- **`streaming/bronze.py`** — Spark Structured Streaming: Kafka → typed columns → Delta, partitioned
  by `event_date`, with the raw payload and Kafka offsets kept for replay. Explicit schema, checked
  against the Pydantic model by a test.
- **`streaming/inspect_bronze.py`** — `make bronze-peek`: row counts, card counts, time range and a
  duplicate check.
- **Docker**: `docker/spark.Dockerfile` (Java 17, jars warmed at build time) and two new Compose
  services — `kafka-setup` (explicit topic creation) and `spark` (runs the Bronze ingest).
- **CI**: a fourth job runs the Spark tests on Temurin 17 with an Ivy cache.

### How to run it

```bash
make up                                  # kafka, topics, spark, redis, mlflow, api
make produce ARGS="--limit 2000"         # replay payments onto the topic
make stream-logs                         # watch the micro-batches land
make bronze-peek                         # summarise the Bronze table

make produce-preview                     # or just look at an event, no Docker needed
```

### Verified

Run on 2026-09-19 in the development container (Spark 3.5.3, Delta 3.2.1):

- `make lint` — clean (28 files).
- Fast suite — **49 passed**.
- Spark suite — **8 passed**, including an end-to-end run of the real streaming writer (file source
  standing in for Kafka) that lands 1000 events in a partitioned Delta table, and a restart test
  proving the checkpoint prevents duplicates.
- Manual end-to-end: 1000 fixture events → Bronze Delta → `inspect_bronze` reported 1000 rows,
  1000 distinct transaction IDs, 44 distinct cards, one `event_date` partition, no duplicates.
- `docker compose --profile core --profile ai config` — valid.

### Not verified yet (needs your laptop)

Still no Docker daemon here, so the Kafka hop itself is untested end to end:

- `make up` building and starting the `spark` and `kafka-setup` containers.
- `make produce` actually publishing to a real broker (the confluent-kafka producer path).
- The Spark job reading from the Kafka source rather than the file source used in tests.

Everything either side of that hop is tested; the hop itself is the gap.

### Known issues / deliberate gaps

- The producer reads the whole CSV into pandas before replaying. Fine for the fixture and for the
  real 590k-row file on 16 GB, but it is not a streaming read.
- `maxOffsetsPerTrigger` is fixed at 20,000. On a slower machine a smaller value may be needed.
- No decision or feature data yet — Bronze is raw events only. Redis and the feature definitions
  arrive in phase 3.

---

## Phase 1 — Foundation ✅

### What is in place

- **Repository structure** for all ten phases, with a package per pipeline stage.
- **`docker-compose.yml`** with Compose profiles (`core`, `ai`; `orchestration` and `dashboard`
  arrive with the code they run) and an explicit `mem_limit` on every service.
  Pinned images: `apache/kafka:3.8.1` (KRaft, no ZooKeeper), `redis:7.4.1-alpine`,
  `ghcr.io/mlflow/mlflow:v2.17.2`, `ollama/ollama:0.5.4`, `pgvector/pgvector:0.8.0-pg16`.
- **`.env.example`** covering every configurable value, and **`common/config.py`**, a single typed
  `Settings` object that reads it. A test fails if the two drift apart.
- **`Makefile`** — `make help` lists everything.
- **`scripts/download_data.sh`** — Kaggle CLI download with clear instructions when the token is
  missing; refuses to re-download if the files are already there.
- **Synthetic dataset generator** (`tests/synthetic.py`) reproducing the exact IEEE-CIS schema
  (394 transaction columns, 41 identity columns), with realistic class imbalance, repeat card
  activity and fraud injected as bursts. `make sample` writes the committed fixture.
- **Scoring API skeleton** — `GET /health`, served by the `api` container.
- **CI** (`.github/workflows/ci.yml`) — ruff, pytest with coverage, a check that the committed
  fixture still matches the generator, and `docker compose config` validation. No Kaggle account,
  no Docker services, no GPU.

### How to run it

```bash
make env        # create .env
make install    # .venv with dev + ml dependencies
make lint       # ruff check + format check
make test       # 21 tests, ~1.5 s
make sample     # regenerate tests/fixtures/
make up         # start kafka, redis, mlflow, api
curl localhost:8000/health
```

### Verified

Run on 2026-09-19 in the development container (Python 3.11.15):

- `make lint` — clean (`ruff 0.7.2`, 18 files).
- `make test` — **21 passed** in 1.36 s.
- `make sample` — 1000 transactions (3.50% fraud, 43 distinct cards), 207 identity rows, 636 KB.
- `docker compose --profile core --profile ai config` — valid.

### Not verified yet (needs your laptop)

The development container has **no Docker daemon**, so nothing below could be executed here:

- `make up` actually starting the containers.
- The `api` image building and serving `/health` from inside Docker.
- Kafka, Redis, MLflow, Ollama and pgvector coming up healthy.

Please run `make up` once and report anything that fails — that is the fastest way to catch an
image or healthcheck problem before phase 2 builds on it.

### Known issues / deliberate gaps

- `orchestration` and `dashboard` profiles are not in the Compose file yet. They arrive in phases 7
  and 9, with the code they run. A profile that starts an empty container is worse than no profile.
- `GET /health` reports `model_loaded: false` unconditionally; there is no model until phase 5.
- No metric in the README is filled in. Nothing has been measured yet.
