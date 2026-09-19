# Progress

What works today, how to run it, and what is known to be missing. Updated at the end of every
phase.

| Phase | Status |
|---|---|
| 1. Foundation | ✅ done |
| 2. Streaming ingestion | ✅ done |
| 3. Features | ⬜ not started |
| 4. Labels and training data | ⬜ not started |
| 5. Model | ⬜ not started |
| 6. Serving | ⬜ not started |
| 7. Orchestration and analytics | ⬜ not started |
| 8. GenAI assistant | ⬜ not started |
| 9. Analyst app and dashboard | ⬜ not started |
| 10. Polish | ⬜ not started |

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
