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
| 7. Orchestration and analytics | ⬜ not started |
| 8. GenAI assistant | ⬜ not started |
| 9. Analyst app and dashboard | ⬜ not started |
| 10. Polish | ⬜ not started |

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
