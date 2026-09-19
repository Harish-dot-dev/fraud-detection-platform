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
	$(VENV)/bin/pytest -m "not slow and not needs_spark and not needs_ollama"

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

bronze-peek: ## Show the last few rows landed in the Bronze Delta table
	$(COMPOSE) run --rm spark python -m streaming.inspect_bronze

##@ Housekeeping

clean: ## Remove caches, build artifacts and the virtualenv
	rm -rf $(VENV) .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean-data: ## Delete generated data (Delta tables, DuckDB, MLflow) - NOT data/raw
	rm -rf data/delta data/warehouse data/mlflow
	@echo "Removed generated data. data/raw was left untouched."

# ---------------------------------------------------------------------------
# Targets below arrive with their phase:
#   features           (phase 3)   build silver + gold feature tables
#   labels             (phase 4)   simulate chargeback arrival
#   train              (phase 5)   train + register the model
#   load-test          (phase 6)   measure /score latency -> reports/latency.json
#   airflow / dbt      (phase 7)
#   llm-eval           (phase 8)   -> reports/llm_eval.json
#   app / dashboard    (phase 9)
#   metrics            (phase 10)  regenerate every reports/*.json
# ---------------------------------------------------------------------------
