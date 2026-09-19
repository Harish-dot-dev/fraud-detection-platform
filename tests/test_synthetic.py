"""Tests for the synthetic dataset generator.

These matter more than they look: every later phase is tested against this
data, so if the generator drifts away from the real IEEE-CIS schema, code that
passes CI would fail on the real files.
"""

from __future__ import annotations

import pandas as pd

from tests.synthetic import (
    IDENTITY_COLUMNS,
    TRANSACTION_COLUMNS,
    generate_transactions,
)


def test_transaction_schema_matches_ieee_cis() -> None:
    """Column names and order must match the real train_transaction.csv."""
    transactions, _ = generate_transactions(n_rows=50, seed=1)

    assert list(transactions.columns) == TRANSACTION_COLUMNS
    # 17 base columns + C1-14 + D1-15 + M1-9 + V1-339
    assert len(TRANSACTION_COLUMNS) == 17 + 14 + 15 + 9 + 339
    assert TRANSACTION_COLUMNS[:4] == [
        "TransactionID",
        "isFraud",
        "TransactionDT",
        "TransactionAmt",
    ]


def test_identity_schema_matches_ieee_cis() -> None:
    """Column names and order must match the real train_identity.csv."""
    _, identity = generate_transactions(n_rows=200, seed=1)

    assert list(identity.columns) == IDENTITY_COLUMNS
    assert IDENTITY_COLUMNS[1] == "id_01"
    assert IDENTITY_COLUMNS[-3:] == ["id_38", "DeviceType", "DeviceInfo"]


def test_generation_is_deterministic() -> None:
    """The same seed must reproduce the fixture byte for byte."""
    first_txn, first_id = generate_transactions(n_rows=100, seed=7)
    second_txn, second_id = generate_transactions(n_rows=100, seed=7)

    pd.testing.assert_frame_equal(first_txn, second_txn)
    pd.testing.assert_frame_equal(first_id, second_id)


def test_different_seeds_produce_different_data() -> None:
    first, _ = generate_transactions(n_rows=100, seed=1)
    second, _ = generate_transactions(n_rows=100, seed=2)

    assert not first["TransactionAmt"].equals(second["TransactionAmt"])


def test_transactions_are_ordered_in_time() -> None:
    """TransactionDT must increase: the producer replays rows in this order."""
    transactions, _ = generate_transactions(n_rows=300, seed=3)

    assert transactions["TransactionDT"].is_monotonic_increasing
    assert transactions["TransactionDT"].is_unique
    assert transactions["TransactionID"].is_unique


def test_class_imbalance_is_realistic() -> None:
    """Fraud should be rare - that imbalance drives most of the design."""
    transactions, _ = generate_transactions(n_rows=2000, seed=4, fraud_rate=0.035)

    fraud_rate = transactions["isFraud"].mean()
    assert 0.02 < fraud_rate < 0.05
    assert set(transactions["isFraud"].unique()) == {0, 1}


def test_identity_covers_only_a_subset_of_transactions() -> None:
    """As in the real data, most transactions have no identity record."""
    transactions, identity = generate_transactions(n_rows=1000, seed=5)

    assert 0 < len(identity) < len(transactions)
    assert identity["TransactionID"].isin(transactions["TransactionID"]).all()
    assert identity["TransactionID"].is_unique


def test_cards_repeat_so_velocity_features_have_signal() -> None:
    """Per-card history is the point of the streaming features."""
    transactions, _ = generate_transactions(n_rows=1000, seed=6)

    # The card identity proxy the pipeline uses (see docs/design_decisions.md).
    card_key = (
        transactions["card1"].astype(str)
        + "|"
        + transactions["addr1"].astype(str)
        + "|"
        + transactions["P_emaildomain"].astype(str)
    )
    assert card_key.nunique() < len(transactions)
    assert card_key.value_counts().max() > 5


def test_v_columns_are_mostly_missing_like_the_real_data() -> None:
    transactions, _ = generate_transactions(n_rows=200, seed=8)

    v_columns = [c for c in transactions.columns if c.startswith("V")]
    missing_share = transactions[v_columns].isna().to_numpy().mean()
    assert missing_share > 0.9
