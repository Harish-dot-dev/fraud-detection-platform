# Progress

What works today, how to run it, and what is known to be missing. Updated at the end of every
phase.

| Phase | Status |
|---|---|
| 1. Foundation | ✅ done |
| 2. Streaming ingestion | ✅ done |
| 3. Features | ✅ done |
| 4. Labels and training data | ✅ done |
| 5. Model | ⬜ not started |
| 6. Serving | ⬜ not started |
| 7. Orchestration and analytics | ⬜ not started |
| 8. GenAI assistant | ⬜ not started |
| 9. Analyst app and dashboard | ⬜ not started |
| 10. Polish | ⬜ not started |

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
