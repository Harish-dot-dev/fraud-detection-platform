# ---------------------------------------------------------------------------
# Friendly entry points for the whole platform. Run `make` or `make help` to
# see everything. Targets are grouped by the phase that introduced them.
# ---------------------------------------------------------------------------

# Use bash so that `set -o pipefail` and [[ ]] behave as expected.
SHELL := /bin/bash
.DEFAULT_GOAL := help

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
COMPOSE := docker compose

.PHONY: help env install lint fmt test test-all sample data up up-ai down ps logs \
	      produce produce-preview topics stream-logs stream-once stream-local bronze-peek \
	      silver gold quality features labels dataset train consume load-test demo-api \
	      export dbt drift decisions-sink warehouse airflow airflow-logs \
	      load-cases llm-eval app publish dashboard readme-metrics metrics \
	      clean clean-data ollama-pull

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "; printf "\nFraud detection platform\n\nUsage: make <target>\n\n"} \
	     /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2} \
	     /^##@/ {printf "\n\033[1m%s\033[0m\n", substr($$0, 5)}' $(MAKEFILE_LIST)
	@echo ""

##@ Setup

env: ## Create .env from .env.example (never overwrites an existing .env)
	@if [ -f .env ]; then \
	@  echo ".env already exists - leaving it alone."; \
	@else \
	@  cp .env.example .env; \
	@  echo "Created .env. Generate a PII salt with:"; \
	@  echo '  python -c "import secrets; print(secrets.token_hex(32))"'; \
	@fi

install: ## Create the local virtualenv and install dev + ml dependencies
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[ml,dev]"
	@echo ""
	@echo "Installed. Heavier groups are opt-in:"
	@echo "  $(PIP) install -e '.[spark]'    # Spark streaming + Delta (needs Java 17)"
	@echo "  $(PIP) install -e '.[genai]'    # embeddings + LangChain (pulls torch, ~2 GB)"
	@echo "  $(PIP) install -e '.[dbt,quality,app]'"

##@ Quality

lint: ## Run ruff (lint + format check)
	$(VENV)/bin/ruff check .
	$(VENV)/bin/ruff format --check .

fmt: ## Auto-fix lint issues and format the code
	$(VENV)/bin/ruff format .
	$(VENV)/bin/ruff check --fix .

test: ## Run the fast test suite (no Spark, no Ollama, no Docker needed)
	$(VENV)/bin/pytest -m "not slow and not needs_spark and not needs_ollama and not needs_pgvector"

test-all: ## Run every test, including the slow and infrastructure-dependent ones
	$(VENV)/bin/pytest

##@ Data

data: ## Download the IEEE-CIS dataset from Kaggle into data/raw/
	./scripts/download_data.sh

sample: ## Regenerate the synthetic CI fixture in tests/fixtures/
	$(PY) scripts/make_synthetic_sample.py --rows 1000 --out tests/fixtures

##@ Stack

up: ## Start the core stack (kafka, redis, mlflow, api)
	$(COMPOSE) --profile core up -d --build
	@echo ""
	@echo "  API      http://localhost:8000/docs"
	@echo "  MLflow   http://localhost:5000"

up-ai: ## Start core + the GenAI services (ollama, pgvector)
	$(COMPOSE) --profile core --profile ai up -d --build

down: ## Stop all containers (named volumes are kept)
	$(COMPOSE) --profile core --profile ai --profile orchestration --profile dashboard down

ps: ## Show container status
	$(COMPOSE) --profile core --profile ai ps

logs: ## Tail logs from all running containers
	$(COMPOSE) --profile core --profile ai logs -f --tail=100

ollama-pull: ## Download the local LLM into the ollama volume (run once)
	$(COMPOSE) exec ollama ollama pull $${OLLAMA_MODEL:-llama3.2:3b}

##@ Pipeline

produce: ## Replay payments onto Kafka. e.g. make produce ARGS="--limit 500 --speedup 0"
	$(PY) -m producer.replay $(ARGS)

produce-preview: ## Print a few payment events without touching Kafka
	$(PY) -m producer.replay --dry-run --limit 3

topics: ## List the Kafka topics
	$(COMPOSE) exec kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server kafka:9092 --list

stream-logs: ## Follow the Bronze streaming job running in Docker
	$(COMPOSE) logs -f spark

stream-once: ## Drain the topic into Bronze Delta and exit (runs in the spark container)
	$(COMPOSE) run --rm spark python -m streaming.bronze --once

stream-local: ## Run the Bronze job from this venv (needs `pip install -e '.[spark]'` + Java 17)
	$(PY) -m streaming.bronze --bootstrap-servers localhost:29092 $(ARGS)

silver: ## Build the Silver layer from Bronze (dedupe, tokenise, quality checks)
	$(COMPOSE) run --rm spark python -m streaming.silver

gold: ## Build the Gold offline feature table from Silver
	$(COMPOSE) run --rm spark python -m streaming.gold

quality: ## Run the Great Expectations suite against Silver -> reports/
	$(COMPOSE) run --rm spark python -m quality.expectations

features: silver gold ## Build Silver and Gold in one go

labels: ## Build the chargeback table (labels, with their real arrival delay)
	$(COMPOSE) run --rm spark python -m training.build_labels

dataset: ## Build a point-in-time training set. e.g. make dataset ARGS="--as-of 2023-04-01"
	$(COMPOSE) run --rm spark python -m training.dataset $(ARGS)

train: ## Train the model, tune thresholds, log to MLflow -> reports/metrics.json
	$(COMPOSE) run --rm spark python -m training.train $(ARGS)

consume: ## Score payments from the topic through the API
	$(PY) -m serving.consumer $(ARGS)

load-test: ## Measure /score latency across concurrency levels -> reports/latency.json
	$(PY) scripts/load_test.py --api-url http://localhost:8000 $(ARGS)

demo-api: ## Run the API with no Docker at all (fakeredis + a model trained in-process)
	$(PY) -m tests.demo_server

export: ## Publish the Delta tables to Parquet for the warehouse
	$(COMPOSE) run --rm spark python -m warehouse.export

dbt: ## Build and test the dbt models on DuckDB
	cd dbt && DUCKDB_PATH=$${DUCKDB_PATH:-../data/warehouse/fraud.duckdb} \
	  WAREHOUSE_EXPORT=$${WAREHOUSE_EXPORT:-../data/warehouse/export} \
	  ../$(VENV)/bin/dbt build --profiles-dir . --project-dir .

drift: ## Evidently drift report -> reports/drift_*.html and drift.json
	$(PY) -m quality.drift

decisions-sink: ## Land the decisions topic in Delta (once)
	$(COMPOSE) run --rm spark python -m streaming.decisions_sink --once

warehouse: export dbt drift ## Export, transform, test and check for drift

airflow: ## Start Airflow (orchestration profile) at http://localhost:8080
	$(COMPOSE) --profile orchestration up -d --build
	@echo "Airflow: http://localhost:8080 (the standalone password is printed in its logs)"

airflow-logs: ## Follow the Airflow logs (the admin password is in here)
	$(COMPOSE) logs -f airflow

load-cases: ## Embed confirmed past cases into pgvector for retrieval
	$(PY) -m genai.load_cases $(ARGS)

llm-eval: ## Evaluate the analyst assistant -> reports/llm_eval.json
	$(PY) -m eval.llm_eval $(ARGS)

app: ## Run the analyst review queue and metrics page
	$(PY) -m streamlit run analyst_app/app.py --server.port 8501

publish: ## Snapshot the warehouse read-only and export Parquet for Power BI
	$(PY) -m warehouse.publish

dashboard: publish ## Start Superset against the read-only snapshot (optional, unverified)
	$(COMPOSE) --profile dashboard up -d
	@echo "Superset: http://localhost:8088 - see dashboard/superset/README.md"

readme-metrics: ## Regenerate the README results table from reports/*.json
	$(PY) scripts/readme_metrics.py

metrics: ## Run every measurement and refresh the README table
	@echo "== quality =="      && $(MAKE) quality
	@echo "== train =="        && $(MAKE) train
	@echo "== drift =="        && $(MAKE) drift
	@echo "== load test =="    && $(PY) scripts/load_test.py --api-url http://localhost:8000
	@echo "== llm eval =="     && $(PY) -m eval.llm_eval
	@$(MAKE) readme-metrics

bronze-peek: ## Show the last few rows landed in the Bronze Delta table
	$(COMPOSE) run --rm spark python -m streaming.inspect_bronze

##@ Housekeeping

clean: ## Remove caches, build artifacts and the virtualenv
	rm -rf $(VENV) .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean-data: ## Delete generated data (Delta tables, DuckDB, MLflow) - NOT data/raw
	rm -rf data/delta data/warehouse data/mlflow
	@echo "Removed generated data. data/raw was left untouched."

