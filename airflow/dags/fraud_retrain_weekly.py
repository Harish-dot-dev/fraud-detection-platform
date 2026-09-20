"""Weekly retrain, with an explicit promotion rule.

    build training set (as of today) ──► train ──► register ──► maybe promote

Two things make this a retrain pipeline rather than a cron job that overwrites
a model file:

**The training set is rebuilt point-in-time.** It is assembled *as of the run
date*, so it contains only payments whose chargebacks had actually arrived by
then. The most recent weeks are deliberately absent - their labels do not exist
yet - and a model trained here has not seen the future.

**A challenger only becomes champion if it earns it.** The new model is
registered every week regardless; the ``champion`` alias moves only if the
challenger beats the incumbent's PR-AUC on the most recent test window. A
retrain that quietly promoted a worse model every Sunday would be worse than
no retrain at all, because nobody would be looking.

Promotion is a registry operation, so the serving API picks the new model up on
its next restart without a deployment.
"""

from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from common import DEFAULT_ARGS, job

with DAG(
    dag_id="fraud_retrain_weekly",
    description="Rebuild the training set point-in-time, retrain, promote only if better",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2023, 1, 1),
    # Sunday night, after the daily batch has landed Saturday's data.
    schedule="0 4 * * 0",
    catchup=False,
    max_active_runs=1,
    tags=["fraud", "ml"],
) as dag:
    start = EmptyOperator(task_id="start")

    # {{ ds }} is the run date: the training set is assembled as of it, not as
    # of "now", so a re-run of an old week reproduces that week's data exactly.
    build_training_set = BashOperator(
        task_id="build_training_set",
        bash_command=job("training.dataset", "--as-of {{ ds }}"),
        doc_md=(
            "Point-in-time training set: a payment is included only once it "
            "has happened *and* its label had arrived by {{ ds }}."
        ),
    )

    train_challenger = BashOperator(
        task_id="train_and_register",
        bash_command=job("training.train"),
        doc_md=(
            "Time-split, scale_pos_weight for the imbalance, thresholds tuned "
            "on validation, evaluated on a test window neither the model nor "
            "the thresholds ever saw. Registers the model and promotes it to "
            "champion only if it beats the incumbent's PR-AUC."
        ),
    )

    # Drift is recomputed after a retrain so the report reflects the model that
    # is now serving rather than the one it replaced.
    refresh_drift = BashOperator(
        task_id="refresh_drift_report",
        bash_command=job("quality.drift"),
    )

    finish = EmptyOperator(task_id="finish")

    start >> build_training_set >> train_challenger >> refresh_drift >> finish
