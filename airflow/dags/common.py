"""Shared helpers for the DAGs.

Every task in this project runs as a **subprocess** (``BashOperator``) rather
than importing the pipeline into the Airflow worker. Three reasons, all
practical:

* The Spark jobs need their own JVM and their own process anyway.
* Airflow's own dependency set is large and opinionated; keeping the platform's
  dependencies out of it means an Airflow upgrade cannot break the model, and a
  pandas upgrade cannot break the scheduler.
* Every job already has a command-line entry point, which is what makes them
  runnable by hand when something goes wrong at 3am. The DAG runs exactly what
  a human would type.
"""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Any

# Where the repository is mounted inside the Airflow container.
PROJECT_ROOT = os.environ.get("FRAUD_PROJECT_ROOT", "/opt/fraud-platform")
# The interpreter that has the platform's dependencies - not Airflow's own.
PYTHON = os.environ.get("FRAUD_PYTHON", "python")

DEFAULT_ARGS: dict[str, Any] = {
    "owner": "fraud-platform",
    "depends_on_past": False,
    # Retries are for transient failures - a broker restarting, a lock. A job
    # that is genuinely broken fails three times and then tells somebody.
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


def job(module: str, args: str = "") -> str:
    """A shell command that runs one of the platform's modules."""
    return f"cd {PROJECT_ROOT} && {PYTHON} -m {module} {args}".strip()


def dbt(command: str = "build") -> str:
    """A dbt command, run from the project directory."""
    return (
        f"cd {PROJECT_ROOT}/dbt && {PYTHON} -m dbt.cli.main {command} "
        f"--profiles-dir . --project-dir ."
    )
