"""Daily batch: land the day's data, rebuild the warehouse, check for drift.

    Bronze ──► Silver ──► Gold ──┐
                  │              ├──► export ──► dbt build ──► drift
    decisions ────┘              │
    chargebacks ─────────────────┘

The order matters in one specific way: **Silver refuses to write a table that
fails its quality suite**, so everything downstream of it is either built on
data that passed the checks or not built at all. A pipeline that carries on
past a failed quality gate and leaves the dashboard showing yesterday's numbers
next to today's date is worse than one that stops.

Chargebacks are reloaded every day on purpose. That is the whole point of the
delayed-label simulation: yesterday's payments have no labels, and the ones
from six weeks ago are only arriving now.
"""

from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from common import DEFAULT_ARGS, dbt, job

with DAG(
    dag_id="fraud_batch_daily",
    description="Bronze to Gold, labels, warehouse and drift monitoring",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2023, 1, 1),
    schedule="0 2 * * *",
    # No backfilling: these jobs rebuild from the current state of the Delta
    # tables rather than processing a date partition, so sixteen simultaneous
    # catch-up runs would fight over the same files to no purpose.
    catchup=False,
    max_active_runs=1,
    tags=["fraud", "batch"],
) as dag:
    start = EmptyOperator(task_id="start")

    # Drain whatever the API has published since the last run.
    land_decisions = BashOperator(
        task_id="land_decisions",
        bash_command=job("streaming.decisions_sink", "--once"),
        doc_md="Kafka decisions topic -> Delta. Analytics must never block scoring.",
    )

    # Bronze -> Silver. Runs the Great Expectations suite and refuses to write
    # if it fails, which is what makes the rest of this DAG trustworthy.
    build_silver = BashOperator(
        task_id="build_silver",
        bash_command=job("streaming.silver"),
        doc_md="Deduplicate, drop non-positive amounts, tokenise, quality-gate.",
    )

    build_gold = BashOperator(
        task_id="build_gold",
        bash_command=job("streaming.gold"),
        doc_md="Offline features, computed with windows that end at each payment.",
    )

    # Today's chargebacks: labels for payments made weeks ago.
    load_chargebacks = BashOperator(
        task_id="load_chargebacks",
        bash_command=job("training.build_labels"),
        doc_md="Reload the label table. Most of today's arrivals are old payments.",
    )

    export_warehouse = BashOperator(
        task_id="export_warehouse",
        bash_command=job("warehouse.export"),
        doc_md="Publish Parquet snapshots so the warehouse never contends with the jobs.",
    )

    # dbt build = run + test. A failing test stops the DAG rather than
    # publishing a dashboard nobody should trust.
    transform_and_test = BashOperator(
        task_id="dbt_build",
        bash_command=dbt("build"),
        doc_md="dbt models and their tests, including the point-in-time label check.",
    )

    monitor_drift = BashOperator(
        task_id="drift_report",
        bash_command=job("quality.drift"),
        doc_md=(
            "Evidently report on features and predictions. Produces a report "
            "for a human, not an alert that blocks the pipeline - drift means "
            "look, not stop."
        ),
    )

    finish = EmptyOperator(task_id="finish")

    start >> [land_decisions, build_silver, load_chargebacks]
    build_silver >> build_gold
    [land_decisions, build_gold, load_chargebacks] >> export_warehouse
    export_warehouse >> transform_and_test >> monitor_drift >> finish
