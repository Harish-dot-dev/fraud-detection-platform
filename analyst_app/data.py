"""Reading the warehouse for the analyst app.

Everything here is read-only against the DuckDB file dbt builds. DuckDB allows
one writer at a time, so the connection is opened read-only and a lock is
reported as a readable message rather than a stack trace: if the daily batch is
mid-run, the right answer is "the warehouse is being rebuilt, try again in a
minute", not a 500.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from common.config import REPO_ROOT, get_settings

logger = logging.getLogger("analyst_app.data")


class WarehouseUnavailable(RuntimeError):
    """The warehouse could not be opened - usually because a job is writing."""


def warehouse_path() -> Path:
    return get_settings().path(get_settings().duckdb_path)


def connect(path: Path | None = None) -> Any:
    """Open the warehouse read-only."""
    import duckdb

    database = path or warehouse_path()
    if not database.exists():
        raise WarehouseUnavailable(f"No warehouse at {database}. Run `make warehouse` to build it.")
    try:
        return duckdb.connect(str(database), read_only=True)
    except Exception as error:  # noqa: BLE001
        raise WarehouseUnavailable(
            "The warehouse is locked - the daily batch is probably rebuilding it. "
            "Try again shortly."
        ) from error


def review_queue(connection: Any, reviewed_ids: set[int], limit: int = 100) -> pd.DataFrame:
    """Payments waiting for a human, most recent first.

    Blocked payments are included as well as reviews: a customer whose payment
    was stopped is waiting on somebody too, and a wrong block is the error that
    generates the complaint.
    """
    frame = connection.execute(
        """
        SELECT transaction_id, decided_at, decision, score, amount, triggered_by,
               rule_name, model_version, top_reason_feature, card_token,
               product_cd, card_brand, device_type, latency_ms
        FROM fct_decisions
        WHERE decision IN ('review', 'block')
        ORDER BY decided_at DESC
        LIMIT ?
        """,
        [limit],
    ).df()

    if reviewed_ids and not frame.empty:
        frame = frame[~frame["transaction_id"].isin(reviewed_ids)]
    return frame


def daily_kpis(connection: Any) -> pd.DataFrame:
    return connection.execute("SELECT * FROM agg_daily_kpis ORDER BY decision_date").df()


def model_performance(connection: Any) -> pd.DataFrame:
    return connection.execute(
        "SELECT * FROM agg_model_performance ORDER BY decision_date, model_version"
    ).df()


def rule_effectiveness(connection: Any) -> pd.DataFrame:
    return connection.execute("SELECT * FROM agg_rule_effectiveness ORDER BY times_fired DESC").df()


def decision_detail(connection: Any, transaction_id: int) -> dict[str, Any] | None:
    """Everything recorded about one decision, straight from the audit log."""
    rows = connection.execute(
        """
        SELECT d.*, c.is_fraud, c.label_available_at
        FROM fct_decisions d
        LEFT JOIN stg_chargebacks c USING (transaction_id)
        WHERE d.transaction_id = ?
        """,
        [transaction_id],
    ).df()
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def decision_features(export_dir: Path, transaction_id: int) -> dict[str, float]:
    """The feature values the decision was made from.

    Read from the decisions snapshot rather than recomputed: the analyst must
    see what the model actually saw, not a re-derivation that might have
    drifted since.
    """
    import duckdb

    rows = (
        duckdb.connect()
        .execute(
            f"""
        SELECT features, top_reasons
        FROM read_parquet('{export_dir}/decisions/*.parquet')
        WHERE transaction_id = ?
        ORDER BY decided_at DESC
        LIMIT 1
        """,
            [transaction_id],
        )
        .fetchall()
    )
    if not rows:
        return {}
    return {"features": dict(rows[0][0] or {}), "top_reasons": list(rows[0][1] or [])}


def default_export_dir() -> Path:
    return REPO_ROOT / "data" / "warehouse" / "export"
