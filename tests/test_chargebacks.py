"""Tests for the simulated chargeback arrival."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from training.chargebacks import (
    LABEL_SOURCE_CHARGEBACK,
    LABEL_SOURCE_MATURED,
    available_as_of,
    build_chargebacks,
    chargeback_delay_days,
    label_summary,
)

MIN_DELAY = 7
MAX_DELAY = 60
MATURITY = 60


@pytest.fixture(scope="module")
def chargebacks(sample_transactions: pd.DataFrame) -> pd.DataFrame:
    return build_chargebacks(sample_transactions)


def test_fraud_is_confirmed_inside_the_dispute_window() -> None:
    delays = [
        chargeback_delay_days(transaction_id, is_fraud=True) for transaction_id in range(1, 500)
    ]

    assert all(MIN_DELAY <= delay <= MAX_DELAY for delay in delays)
    # The delays should spread across the window, not clump at one value.
    assert max(delays) - min(delays) > 40


def test_legitimate_payments_are_only_confirmed_once_the_window_closes() -> None:
    """Nothing ever arrives to say a payment was fine; time is what says it."""
    assert chargeback_delay_days(1, is_fraud=False) == float(MATURITY)
    assert chargeback_delay_days(999, is_fraud=False) == float(MATURITY)


def test_the_delay_is_stable_for_a_given_transaction() -> None:
    assert chargeback_delay_days(12345, is_fraud=True) == chargeback_delay_days(
        12345, is_fraud=True
    )


def test_the_delay_does_not_depend_on_what_was_processed_before_it(
    sample_transactions: pd.DataFrame,
) -> None:
    """The property that makes daily incremental loading safe.

    Airflow loads this table a day at a time. If a payment's delay depended on
    a running RNG, the timeline would change every time the table was rebuilt
    and every point-in-time guarantee with it.
    """
    full = build_chargebacks(sample_transactions).set_index("transaction_id")
    chunk = build_chargebacks(sample_transactions.tail(100)).set_index("transaction_id")

    pd.testing.assert_series_equal(
        full.loc[chunk.index, "label_available_at"], chunk["label_available_at"]
    )


def test_a_different_seed_produces_a_different_timeline() -> None:
    assert chargeback_delay_days(12345, is_fraud=True, seed=1) != chargeback_delay_days(
        12345, is_fraud=True, seed=2
    )


def test_label_available_at_is_the_payment_plus_the_delay(chargebacks: pd.DataFrame) -> None:
    row = chargebacks.iloc[0]

    assert row["label_available_at"] == row["event_time"] + timedelta(days=row["delay_days"])
    assert row["label_available_at"] > row["event_time"]


def test_label_source_records_how_the_label_was_learned(chargebacks: pd.DataFrame) -> None:
    fraud = chargebacks[chargebacks["is_fraud"] == 1]
    legitimate = chargebacks[chargebacks["is_fraud"] == 0]

    assert set(fraud["label_source"]) == {LABEL_SOURCE_CHARGEBACK}
    assert set(legitimate["label_source"]) == {LABEL_SOURCE_MATURED}


def test_every_transaction_gets_exactly_one_label(
    chargebacks: pd.DataFrame, sample_transactions: pd.DataFrame
) -> None:
    assert len(chargebacks) == len(sample_transactions)
    assert chargebacks["transaction_id"].is_unique


def test_on_the_day_of_the_last_payment_almost_nothing_is_known(
    chargebacks: pd.DataFrame,
) -> None:
    """The headline fact this whole module exists to express.

    The fixture covers half a day, so on the day the last payment lands, not a
    single dispute has come back yet. On the real six-month dataset the same
    effect leaves the most recent two months unusable.
    """
    summary = label_summary(chargebacks, chargebacks["event_time"].max())

    assert summary["labels_known"] == 0
    assert summary["labels_pending"] == len(chargebacks)


def test_labels_become_known_as_time_passes(chargebacks: pd.DataFrame) -> None:
    last_payment = chargebacks["event_time"].max()

    known_after_30_days = available_as_of(chargebacks, last_payment + timedelta(days=30))
    known_after_61_days = available_as_of(chargebacks, last_payment + timedelta(days=61))

    # At 30 days only chargebacks have come in, so everything known is fraud.
    assert set(known_after_30_days["is_fraud"]) == {1}
    assert len(known_after_61_days) == len(chargebacks)


def test_the_source_file_must_actually_have_labels() -> None:
    with pytest.raises(ValueError, match="isFraud"):
        build_chargebacks(pd.DataFrame({"TransactionID": [1], "TransactionDT": [86400]}))
