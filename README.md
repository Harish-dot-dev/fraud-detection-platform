# Real-Time Payment Fraud Detection Platform

Payments stream in through Kafka, live behavioural features are computed with Spark Structured
Streaming and cached in Redis, and a FastAPI service combines a rules engine with an XGBoost model
to return **allow / review / block**. Flagged payments go to a Streamlit review queue where a local
LLM writes a grounded case summary using retrieval over past confirmed cases. Delayed chargeback
labels feed scheduled retraining, drift monitoring and dashboards. Everything runs locally, for
free, with `docker compose up`.

> **Status: phase 7 of 10 complete.** See [PROGRESS.md](PROGRESS.md) for exactly what works today.
> No performance numbers are published yet because none have been measured yet — see
> [Results](#results).

---

## Architecture

```mermaid
flowchart LR
    subgraph ingest["Ingestion"]
        P["Kafka producer<br/>replays IEEE-CIS in time order"]
        K(["topic: payments"])
        P --> K
    end

    subgraph stream["Stream processing"]
        S["Spark Structured Streaming<br/>windowed per-card features"]
        R[("Redis<br/>online feature store")]
        B[("Delta Bronze<br/>raw events")]
        K --> S --> R
        S --> B
    end

    subgraph serve["Real-time scoring"]
        C["Scoring consumer"]
        API["FastAPI /score<br/>rules → XGBoost → SHAP"]
        D(["topic: decisions"])
        DL[("Delta decisions<br/>audit log")]
        K --> C --> API
        R --> API
        API --> D
        API --> DL
    end

    subgraph batch["Batch & orchestration (Airflow)"]
        SI[("Delta Silver<br/>clean + PII tokenised")]
        G[("Delta Gold<br/>offline features")]
        CB[("Chargebacks<br/>delayed labels")]
        DBT["dbt on DuckDB<br/>KPIs + tests"]
        TR["Weekly retrain<br/>champion / challenger"]
        EV["Evidently<br/>drift reports"]
        B --> SI --> G --> TR
        CB --> TR
        G --> DBT
        DL --> DBT
        TR --> EV
    end

    subgraph ai["Analyst assistant"]
        PG[("Postgres + pgvector<br/>past confirmed cases")]
        LLM["Ollama llama3.2:3b<br/>structured JSON summary"]
        ST["Streamlit review queue"]
        DL --> LLM
        PG --> LLM
        LLM --> ST
        ST -->|analyst decision| PG
    end

    MR[("MLflow registry")] --> API
    TR --> MR
    DBT --> DASH["Dashboard"]
```

Full walkthrough: [docs/architecture.md](docs/architecture.md).
Why each piece was chosen: [docs/design_decisions.md](docs/design_decisions.md).

---

## Quick start

### Prerequisites

| Requirement | Notes |
|---|---|
| Docker Desktop / Engine | Allocate **at least 8 GB** of RAM to Docker |
| Python 3.11 | For the host-side tooling (`make install`) |
| Java 17 | Only needed to run the Spark jobs outside Docker |
| Kaggle account | Only needed for the real dataset (see below) |

```bash
make env        # create .env from .env.example
make install    # create .venv and install dev + ml dependencies
make test       # run the test suite (no Docker, no dataset needed)
make up         # start kafka, redis, mlflow and the scoring API
```

Then open the API at <http://localhost:8000/docs> and MLflow at <http://localhost:5000>.
`make down` stops everything.

### Send some payments through it

```bash
make produce ARGS="--limit 2000"   # replay transactions onto the Kafka topic
make stream-logs                   # watch Spark land them in Bronze Delta
make bronze-peek                   # row counts, card counts, time range
```

Then build the batch layers and train a model:

```bash
make features                      # Bronze -> Silver (tokenised) -> Gold (offline features)
make quality                       # Great Expectations suite -> reports/quality_silver.json
make labels                        # chargebacks, with their real arrival delay
make dataset                       # point-in-time training set
make train                         # -> reports/metrics.json, reports/thresholds.json
```

Then build the warehouse and the dashboards' data:

```bash
make warehouse                     # export -> dbt build (7 models, 33 tests) -> drift report
make airflow                       # or orchestrate all of it: http://localhost:8080
```

Then score payments through the API:

```bash
make consume                       # payments topic -> POST /score
make load-test                     # -> reports/latency.json
curl -s localhost:8000/health | jq
```

`make demo-api` runs the whole scoring API with no Docker at all — fakeredis and a model trained
in-process — which is the quickest way to see a decision without setting anything up.

`make produce-preview` prints a couple of payment events without needing Docker at all — the
quickest way to see what flows through the system.

### Generate a PII salt

The Silver layer tokenises the card identity with a salted hash. The example salt is a placeholder
and the API reports `pii_salt_configured: false` until you replace it:

```bash
python -c "import secrets; print(secrets.token_hex(32))"   # paste into PII_HASH_SALT in .env
```

### Get the dataset

The IEEE-CIS Fraud Detection dataset is competition data and is **not** committed to this
repository. You need to accept the competition rules and use your own API token:

1. Accept the rules at <https://www.kaggle.com/competitions/ieee-fraud-detection/rules>
2. Create a token at <https://www.kaggle.com/settings> and save it to `~/.kaggle/kaggle.json`
3. `make data`

Or download the ZIP by hand and unzip `train_transaction.csv` and `train_identity.csv` into
`data/raw/`.

**You do not need the dataset to run the tests.** `tests/fixtures/` holds a committed synthetic
sample with exactly the same schema, generated by `make sample`.

### Low-memory mode

Compose profiles let you run a subset of the stack:

| Profile | Services | Approx. RAM | Command |
|---|---|---|---|
| `core` | kafka, redis, mlflow, api | ~4.0 GB | `make up` |
| `ai` | ollama, pgvector | ~4.5 GB | `make up-ai` (adds to core) |
| `orchestration` | airflow (standalone) | ~3.0 GB | `make airflow` |
| `dashboard` | superset | phase 9 | — |

On a 16 GB laptop, `core` + `ai` is the intended maximum. Every container has an explicit
`mem_limit` in `docker-compose.yml`.

---

## Results

**Nothing here is filled in from an estimate.** Every number in this table comes from a script that
writes a JSON file into `reports/`, and the table is populated from those files. Where a row says
*not yet measured*, the phase that produces it has not been built or has not been run against the
real dataset.

| Metric | Value | Source |
|---|---|---|
| PR-AUC (test window) | *not yet measured* | `make train` → `reports/metrics.json` |
| Precision @ block threshold | *not yet measured* | `make train` → `reports/metrics.json` |
| Recall @ block threshold | *not yet measured* | `make train` → `reports/metrics.json` |
| Fraud value caught vs missed | *not yet measured* | `make train` → `reports/metrics.json` |
| False positive rate | *not yet measured* | `make train` → `reports/metrics.json` |
| Chosen thresholds + cost rationale | *not yet measured* | `make train` → `reports/thresholds.json` |
| `/score` p50 / p95 / p99 latency | 12.7 / 20.3 / 21.4 ms *(native services, not containers — see PROGRESS.md)* | `make load-test` → `reports/latency.json` |
| LLM factual accuracy / schema validity | *not yet measured* | `reports/llm_eval.json` (phase 8) |
| Retrieval quality (label match) | *not yet measured* | `reports/llm_eval.json` (phase 8) |

The "under 100 ms" scoring target is a **goal**, not a claim. The measured number will be published
here once `make load-test` has been run against the full stack, whatever it turns out to be.

What has been measured so far, against **real Redis, a real MLflow registry and a real Kafka
broker** (all running natively on one four-core host, not in containers): **p50 12.7 ms, p95
20.3 ms, p99 21.4 ms** unqueued, saturating at ~71 req/s in a single process. Full table and
conditions in [PROGRESS.md](PROGRESS.md). The docker-compose number will differ and is not being
guessed at here.

---

## Why accuracy is the wrong metric here

Fraud is about 3.5% of transactions in this dataset. A model that predicts "not fraud" for every
payment is 96.5% accurate and catches nothing. That is why this project reports **PR-AUC** plus
precision and recall at the chosen operating points, and converts errors into money using an
explicit cost model rather than treating a missed fraud and a wrongly blocked customer as equally
bad.

More on this, in plain English, in [docs/interview_notes.md](docs/interview_notes.md).

---

## Repository layout

```
├── common/          shared configuration (one typed Settings object)
├── producer/        Kafka producer: replays transactions in time order
├── streaming/       Spark Structured Streaming jobs
├── features/        feature definitions shared by online and offline paths
├── rules/           YAML rules engine configuration
├── serving/         FastAPI scoring service
├── training/        dataset building, training, thresholds, SHAP
├── airflow/dags/    orchestration
├── dbt/             DuckDB transformations and tests
├── quality/         Great Expectations suites
├── genai/           embeddings, retrieval, LLM summaries
├── analyst_app/     Streamlit review queue
├── dashboard/       Superset config + a Power BI folder for .pbix exports
├── eval/            LLM evaluation harness
├── scripts/         data download, fixture generation, load test
├── tests/           test suite + synthetic data generator + fixtures
├── reports/         generated metrics (gitignored except .gitkeep)
└── docs/            architecture, design decisions, interview notes
```

---

## Screenshots

*Placeholders — to be added once the analyst app and dashboard are built (phase 9).*

| Analyst review queue | Dashboard |
|---|---|
| _screenshot pending_ | _screenshot pending_ |

---

## Limitations

Tracked honestly in [docs/design_decisions.md](docs/design_decisions.md); the short version is that
this is a laptop-scale simulation of a production system. The dataset has no real card identifier
(a proxy is used), transaction time is synthetic, and the local 3B LLM is far weaker than what a
real fraud team would deploy.

## Licence

Code: see [LICENSE](LICENSE). The IEEE-CIS dataset is subject to the competition rules and is not
redistributed here.
