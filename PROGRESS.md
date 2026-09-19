# Progress

What works today, how to run it, and what is known to be missing. Updated at the end of every
phase.

| Phase | Status |
|---|---|
| 1. Foundation | ✅ done |
| 2. Streaming ingestion | ⬜ not started |
| 3. Features | ⬜ not started |
| 4. Labels and training data | ⬜ not started |
| 5. Model | ⬜ not started |
| 6. Serving | ⬜ not started |
| 7. Orchestration and analytics | ⬜ not started |
| 8. GenAI assistant | ⬜ not started |
| 9. Analyst app and dashboard | ⬜ not started |
| 10. Polish | ⬜ not started |

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
