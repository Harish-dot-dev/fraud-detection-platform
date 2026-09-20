# Architecture

The end-to-end diagram lives in [the README](../README.md#architecture). This document goes
component by component and is filled in as each phase lands.

## Components

| Component | Phase | Status |
|---|---|---|
| Kafka producer (`producer/`) | 2 | done |
| Spark Structured Streaming (`streaming/`) | 2–3 | done (Bronze + online features) |
| Shared feature definitions (`features/`) | 3 | done (online + offline, consistency-tested) |
| Medallion layers (Bronze / Silver / Gold) | 2–3 | done |
| Chargeback label simulation | 4 | done |
| Point-in-time training set builder (`training/`) | 4 | done (leakage-tested) |
| XGBoost model + MLflow registry | 5 | done (time split, cost-tuned thresholds, SHAP) |
| Rules engine (`rules/`) | 6 | done |
| Scoring API (`serving/`) | 6 | done (`/score`: rules + model + SHAP + audit log) |
| Airflow DAGs (`airflow/dags/`) | 7 | done (both DAGs run green end to end) |
| dbt models (`dbt/`) | 7 | done (7 models, 33 tests) |
| GenAI assistant (`genai/`) | 8 | done (pgvector retrieval, grounded summaries, eval harness) |
| Analyst app (`analyst_app/`) | 9 | not started |
| Dashboard (`dashboard/`) | 9 | not started |

## Data flow in one paragraph

A producer replays historical transactions onto a Kafka topic in their original time order, sped up
by a configurable factor. A Spark Structured Streaming job consumes that topic, writes the raw
events to a Bronze Delta table, and maintains per-card rolling aggregates in Redis. A scoring
consumer calls the FastAPI service, which reads those aggregates, applies the current transaction on
top, runs a YAML-configured rules engine, scores the result with an XGBoost model loaded from the
MLflow registry, and returns `allow`, `review` or `block` with the reasons behind it. Every decision
is appended to an audit log. Batch jobs promote Bronze to Silver (cleaned, deduplicated, PII
tokenised, quality-checked) and then to Gold (offline features), simulate the delayed arrival of
chargeback labels, and rebuild the training set with point-in-time correctness. A weekly retrain
registers a challenger and promotes it only if it beats the champion on the most recent test window.

## Infrastructure

See `docker-compose.yml`. Profiles and memory budgets are documented in
[the README](../README.md#low-memory-mode).
