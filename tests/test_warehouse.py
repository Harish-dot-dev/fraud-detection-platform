"""Tests for the dbt warehouse.

These build the whole project - models and tests - against a tiny fixture
warehouse written straight to Parquet, so the SQL is exercised without needing
Spark or the full pipeline. The numbers are chosen so the assertions can be
read as statements about fraud rather than about SQL.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DBT_DIR = REPO_ROOT / "dbt"

pytestmark = pytest.mark.needs_dbt

NOW = datetime(2023, 6, 1, 12, 0, tzinfo=UTC)


def _fixture_warehouse(export_dir: Path) -> None:
    """Six payments with a known, hand-checkable outcome each."""
    payments = []
    decisions = []
    labels = []

    # (decision, is_fraud, amount, label matured?)
    cases = [
        ("block", 1, 500.0, True),  # fraud_blocked
        ("review", 1, 300.0, True),  # fraud_to_review
        ("allow", 1, 900.0, True),  # fraud_missed - the expensive one
        ("block", 0, 50.0, True),  # false_block
        ("review", 0, 40.0, True),  # false_review
        ("allow", 0, 30.0, True),  # correctly_allowed
        ("block", 1, 700.0, False),  # pending: label has not arrived
    ]

    for index, (decision, is_fraud, amount, matured) in enumerate(cases):
        transaction_id = 1000 + index
        payments.append(
            {
                "transaction_id": transaction_id,
                "card_token": f"{index:032d}",
                "event_time": NOW,
                "amount": amount,
                "product_cd": "W",
                "card4": "visa",
                "card6": "debit",
                "p_emaildomain": "gmail.com",
                "r_emaildomain": "gmail.com",
                "device_type": "desktop",
            }
        )
        decisions.append(
            {
                "transaction_id": transaction_id,
                "card_token": f"{index:032d}",
                "decided_at": NOW,
                "decision": decision,
                "score": 0.9 if decision == "block" else 0.4 if decision == "review" else 0.01,
                "triggered_by": "model",
                "reason": "test",
                "rule_name": "amount_over_hard_limit" if index == 0 else None,
                "model_version": "1",
                "review_threshold": 0.3,
                "block_threshold": 0.8,
                "latency_ms": 10.0 + index,
                "degraded": False,
                "top_reasons": [] if decision == "allow" else [{"feature": "amount"}],
            }
        )
        labels.append(
            {
                "transaction_id": transaction_id,
                "is_fraud": is_fraud,
                "event_time": NOW,
                # An unmatured label is dated far in the future.
                "label_available_at": NOW - timedelta(days=1)
                if matured
                else NOW + timedelta(days=3650),
                "label_source": "chargeback" if is_fraud else "matured",
                "delay_days": 30.0,
            }
        )

    for name, rows in (
        ("silver", payments),
        ("decisions", decisions),
        ("chargebacks", labels),
        ("gold", payments),
    ):
        directory = export_dir / name
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(directory / "part-0.parquet", index=False)


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    """Build the dbt project against the fixture data."""
    root = tmp_path_factory.mktemp("warehouse")
    export_dir = root / "export"
    database = root / "fraud.duckdb"
    _fixture_warehouse(export_dir)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dbt.cli.main",
            "build",
            "--profiles-dir",
            ".",
            "--project-dir",
            ".",
        ],
        cwd=DBT_DIR,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(root),
            "DUCKDB_PATH": str(database),
            "WAREHOUSE_EXPORT": str(export_dir),
            "DBT_TARGET_PATH": str(root / "target"),
        },
    )
    assert result.returncode == 0, result.stdout[-4000:]
    return duckdb.connect(str(database), read_only=True)


def test_dbt_models_and_tests_all_pass(warehouse) -> None:
    """`dbt build` runs the models *and* their tests; a failure fails this."""
    tables = {row[0] for row in warehouse.execute("SHOW TABLES").fetchall()}

    assert {"fct_decisions", "agg_daily_kpis", "agg_model_performance"} <= tables


def test_each_payment_appears_once(warehouse) -> None:
    rows = warehouse.execute(
        "SELECT count(*), count(distinct transaction_id) FROM fct_decisions"
    ).fetchone()

    assert rows[0] == rows[1] == 7


def test_outcomes_are_classified_correctly(warehouse) -> None:
    outcomes = dict(
        warehouse.execute("SELECT outcome, count(*) FROM fct_decisions GROUP BY 1").fetchall()
    )

    assert outcomes["fraud_blocked"] == 1
    assert outcomes["fraud_to_review"] == 1
    assert outcomes["fraud_missed"] == 1
    assert outcomes["false_block"] == 1
    assert outcomes["correctly_allowed"] == 1
    # The payment whose chargeback has not arrived is not scored as anything.
    assert outcomes["pending"] == 1


def test_a_label_that_has_not_arrived_is_not_used(warehouse) -> None:
    """Point-in-time correctness, in the warehouse this time.

    The seventh payment is fraud and was blocked - but its chargeback is dated
    ten years out, so today's dashboard must not count it as a win.
    """
    row = warehouse.execute(
        "SELECT is_confirmed_fraud, label_is_known, outcome FROM fct_decisions "
        "WHERE transaction_id = 1006"
    ).fetchone()

    assert row[0] is None
    assert row[1] is False
    assert row[2] == "pending"


def test_daily_kpis_separate_volumes_from_outcomes(warehouse) -> None:
    row = warehouse.execute(
        "SELECT payments_scored, sent_to_review, blocked, labels_known, labels_pending, "
        "fraud_missed, false_blocks, false_positive_rate FROM agg_daily_kpis"
    ).fetchone()

    assert row[0] == 7  # scored
    assert row[1] == 2  # reviewed
    assert row[2] == 3  # blocked
    assert row[3] == 6  # labels known
    assert row[4] == 1  # still pending
    assert row[5] == 1  # fraud missed
    assert row[6] == 1  # false blocks
    # One false block out of three known-legitimate payments.
    assert row[7] == pytest.approx(1 / 3, abs=1e-6)


def test_the_expensive_miss_shows_up_in_value_not_just_count(warehouse) -> None:
    """Counting frauds caught would call this a good day. The money says otherwise."""
    caught, missed = warehouse.execute(
        "SELECT fraud_value_caught, fraud_value_missed FROM agg_daily_kpis"
    ).fetchone()

    assert caught == pytest.approx(800.0)  # 500 blocked + 300 reviewed
    assert missed == pytest.approx(900.0)  # the single allowed fraud
    assert missed > caught


def test_model_performance_uses_matured_labels_only(warehouse) -> None:
    row = warehouse.execute(
        "SELECT labelled_payments, precision_at_block, recall_at_block, recall_including_review "
        "FROM agg_model_performance WHERE model_version = '1'"
    ).fetchone()

    # Six matured payments: three fraud (blocked, reviewed, allowed) and three
    # legitimate (blocked, reviewed, allowed).
    assert row[0] == 6  # the pending payment is excluded
    assert row[1] == pytest.approx(0.5)  # 1 of the 2 blocks was fraud
    assert row[2] == pytest.approx(1 / 3)  # 1 of 3 known frauds was blocked
    # Counting the review queue, 2 of the 3 frauds reached a human.
    assert row[3] == pytest.approx(2 / 3)


def test_rule_effectiveness_reports_a_hit_rate(warehouse) -> None:
    row = warehouse.execute(
        "SELECT rule_name, times_fired, caught_fraud, hit_rate FROM agg_rule_effectiveness"
    ).fetchone()

    assert row[0] == "amount_over_hard_limit"
    assert row[1] == 1
    assert row[2] == 1
    assert row[3] == pytest.approx(1.0)
