"""Tests for the Kafka payment event schema."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common.events import (
    CARD_KEY_MISSING,
    PaymentEvent,
    card_identity_proxy,
    payment_event_from_row,
)
from common.timeline import DEFAULT_EPOCH


def test_the_event_carries_no_label() -> None:
    """The single most important test in this file.

    At authorisation time nobody knows whether a payment is fraudulent - the
    chargeback arrives weeks later. Keeping isFraud out of the event means the
    streaming and scoring paths cannot leak it even by accident.
    """
    assert "is_fraud" not in PaymentEvent.model_fields
    assert "isFraud" not in PaymentEvent.model_fields


def test_card_key_is_built_from_the_documented_proxy() -> None:
    assert card_identity_proxy(13926, 315.0, "gmail.com") == "13926|315|gmail.com"


def test_card_key_renders_whole_floats_as_integers() -> None:
    """card1/addr1 arrive as floats because of NaNs; 315.0 is a bad key."""
    assert card_identity_proxy(13926.0, 315.0, "gmail.com") == "13926|315|gmail.com"


def test_card_key_handles_missing_components() -> None:
    key = card_identity_proxy(13926, np.nan, None)

    assert key == f"13926|{CARD_KEY_MISSING}|{CARD_KEY_MISSING}"


def test_event_is_built_from_a_transaction_row(sample_transactions: pd.DataFrame) -> None:
    row = sample_transactions.iloc[0].to_dict()

    event = payment_event_from_row(row, epoch=DEFAULT_EPOCH)

    assert event.transaction_id == int(row["TransactionID"])
    assert event.amount == float(row["TransactionAmt"])
    assert event.transaction_dt == int(row["TransactionDT"])
    assert event.event_time == DEFAULT_EPOCH + pd.Timedelta(seconds=int(row["TransactionDT"]))
    assert event.card_key.startswith(str(int(row["card1"])))


def test_missing_values_become_none_not_nan(sample_transactions: pd.DataFrame) -> None:
    """NaN would serialise to invalid JSON and break the Spark parser."""
    rows = sample_transactions[sample_transactions["dist2"].isna()]
    event = payment_event_from_row(rows.iloc[0].to_dict())

    assert event.dist2 is None
    assert "NaN" not in event.model_dump_json()


def test_sparse_blocks_only_carry_populated_columns(sample_transactions: pd.DataFrame) -> None:
    row = sample_transactions.iloc[0].to_dict()

    event = payment_event_from_row(row)

    # 339 V columns exist; only the populated ones travel.
    assert 0 < len(event.vesta) < 339
    assert all(not np.isnan(v) for v in event.vesta.values())
    assert len(event.counts) == 14


def test_identity_is_split_by_type(
    sample_transactions: pd.DataFrame, sample_identity: pd.DataFrame
) -> None:
    """Numeric and categorical identity fields are separate maps.

    A single map<string,string> would force every id_01 score to be stringified
    and cast back downstream.
    """
    identity_row = sample_identity.iloc[0].to_dict()
    transaction_id = int(identity_row["TransactionID"])
    transaction = sample_transactions[sample_transactions["TransactionID"] == transaction_id].iloc[
        0
    ]

    event = payment_event_from_row(transaction.to_dict(), identity_row)

    assert "id_01" in event.identity_numeric
    assert isinstance(event.identity_numeric["id_01"], float)
    assert "id_12" in event.identity_categorical
    assert event.device_type in {"desktop", "mobile"}


def test_event_without_identity_has_empty_device_fields(
    sample_transactions: pd.DataFrame,
) -> None:
    event = payment_event_from_row(sample_transactions.iloc[0].to_dict(), identity=None)

    assert event.identity_numeric == {}
    assert event.device_type is None


def test_json_round_trip(sample_transactions: pd.DataFrame) -> None:
    event = payment_event_from_row(sample_transactions.iloc[5].to_dict())

    restored = PaymentEvent.model_validate(json.loads(event.model_dump_json()))

    assert restored == event
