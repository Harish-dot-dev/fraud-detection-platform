"""The payment event that flows through Kafka.

One schema, defined once, used by the producer, the Bronze streaming job and
the scoring API. If these three disagree about a field name, the pipeline
breaks in a way that is tedious to debug, so the Spark schema in
``streaming/bronze.py`` is checked against this model by a test.

Two deliberate design choices, both worth explaining in an interview:

**The label is not in the event.** A real payment message cannot contain
``isFraud`` - at authorisation time nobody knows. Confirmation arrives weeks
later as a chargeback (phase 4). Leaving the column out of the event entirely
means the streaming and scoring paths *cannot* leak it, by construction rather
than by discipline. ``test_events.py`` asserts this.

**Sparse blocks are maps, not columns.** The 339 V columns are ~90% empty. JSON
with 339 nulls per message is wasteful, and a 339-column Spark schema is
unreadable, so the populated ones travel as a map.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from common.timeline import DEFAULT_EPOCH, to_event_time

# Column groups in the source data.
COUNT_COLUMNS = [f"C{i}" for i in range(1, 15)]
DELTA_COLUMNS = [f"D{i}" for i in range(1, 16)]
MATCH_COLUMNS = [f"M{i}" for i in range(1, 10)]
VESTA_COLUMNS = [f"V{i}" for i in range(1, 340)]
IDENTITY_NUMERIC_COLUMNS = [f"id_{i:02d}" for i in range(1, 12)]
IDENTITY_CATEGORICAL_COLUMNS = [f"id_{i:02d}" for i in range(12, 39)]

CARD_KEY_SEPARATOR = "|"
CARD_KEY_MISSING = "unknown"


def card_identity_proxy(card1: Any = None, addr1: Any = None, p_emaildomain: Any = None) -> str:
    """Build the card identity proxy used as the Kafka partition key.

    The dataset has no card identifier, so the platform approximates one with
    ``card1 + addr1 + P_emaildomain`` - a common approach for this dataset and a
    documented approximation, not ground truth (docs/design_decisions.md).

    Using it as the Kafka key matters: Kafka guarantees ordering *within a
    partition*, so keying by card keeps one card's payments in order. Velocity
    features would be wrong if two payments on the same card were processed out
    of sequence.
    """
    parts = []
    for value in (card1, addr1, p_emaildomain):
        cleaned = _clean(value)
        if cleaned is None:
            parts.append(CARD_KEY_MISSING)
        elif isinstance(cleaned, float) and cleaned.is_integer():
            # card1 and addr1 arrive from pandas as floats (because of NaNs);
            # "12345" is a far better key than "12345.0".
            parts.append(str(int(cleaned)))
        else:
            parts.append(str(cleaned))
    return CARD_KEY_SEPARATOR.join(parts)


class PaymentEvent(BaseModel):
    """A single payment as the platform sees it at authorisation time."""

    # --- Identifiers and time ---
    transaction_id: int
    # Derived from transaction_dt via the synthetic timeline (common/timeline.py).
    event_time: datetime
    # The original offset is kept so any row can be traced back to the source.
    transaction_dt: int
    # Partition key: the card identity proxy.
    card_key: str

    # --- Payment ---
    amount: float
    product_cd: str | None = None
    card1: int | None = None
    card2: float | None = None
    card3: float | None = None
    card4: str | None = None
    card5: float | None = None
    card6: str | None = None
    addr1: float | None = None
    addr2: float | None = None
    dist1: float | None = None
    dist2: float | None = None
    p_emaildomain: str | None = None
    r_emaildomain: str | None = None

    # --- Sparse blocks, carried as maps ---
    # C1-C14: counts of addresses, phone numbers and emails linked to the card.
    counts: dict[str, float] = Field(default_factory=dict)
    # D1-D15: "days since" timedeltas.
    deltas: dict[str, float] = Field(default_factory=dict)
    # M1-M9: match flags (name on card matches, address matches, ...).
    match_flags: dict[str, str] = Field(default_factory=dict)
    # V1-V339: Vesta's own engineered features; only the populated ones travel.
    vesta: dict[str, float] = Field(default_factory=dict)

    # --- Device / identity, present for roughly a quarter of transactions ---
    identity_numeric: dict[str, float] = Field(default_factory=dict)
    identity_categorical: dict[str, str] = Field(default_factory=dict)
    device_type: str | None = None
    device_info: str | None = None

    # NOTE: there is deliberately no `is_fraud` field. See the module docstring.


def _clean(value: Any) -> Any:
    """Normalise a pandas cell: NaN / NaT / empty string become ``None``."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    # pandas.NA and numpy NaT do not compare equal to themselves.
    if value != value:  # noqa: PLR0124 - the standard NaN check
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _float_map(row: Mapping[str, Any], columns: list[str]) -> dict[str, float]:
    """Collect the populated numeric columns of a block into a map."""
    out: dict[str, float] = {}
    for column in columns:
        value = _clean(row.get(column))
        if value is not None:
            out[column] = float(value)
    return out


def _string_map(row: Mapping[str, Any], columns: list[str]) -> dict[str, str]:
    """Collect the populated categorical columns of a block into a map."""
    out: dict[str, str] = {}
    for column in columns:
        value = _clean(row.get(column))
        if value is not None:
            out[column] = str(value)
    return out


def _optional_float(row: Mapping[str, Any], column: str) -> float | None:
    value = _clean(row.get(column))
    return None if value is None else float(value)


def _optional_str(row: Mapping[str, Any], column: str) -> str | None:
    value = _clean(row.get(column))
    return None if value is None else str(value)


def payment_event_from_row(
    transaction: Mapping[str, Any],
    identity: Mapping[str, Any] | None = None,
    epoch: datetime = DEFAULT_EPOCH,
) -> PaymentEvent:
    """Build a :class:`PaymentEvent` from one IEEE-CIS transaction row.

    Args:
        transaction: a row of ``train_transaction.csv`` (or the fixture).
        identity: the matching ``train_identity.csv`` row, if there is one.
        epoch: reference date the ``TransactionDT`` offset is mapped onto.
    """
    identity = identity or {}
    card1 = _clean(transaction.get("card1"))

    return PaymentEvent(
        transaction_id=int(transaction["TransactionID"]),
        transaction_dt=int(transaction["TransactionDT"]),
        event_time=to_event_time(transaction["TransactionDT"], epoch),
        card_key=card_identity_proxy(
            card1=transaction.get("card1"),
            addr1=transaction.get("addr1"),
            p_emaildomain=transaction.get("P_emaildomain"),
        ),
        amount=float(transaction["TransactionAmt"]),
        product_cd=_optional_str(transaction, "ProductCD"),
        card1=None if card1 is None else int(card1),
        card2=_optional_float(transaction, "card2"),
        card3=_optional_float(transaction, "card3"),
        card4=_optional_str(transaction, "card4"),
        card5=_optional_float(transaction, "card5"),
        card6=_optional_str(transaction, "card6"),
        addr1=_optional_float(transaction, "addr1"),
        addr2=_optional_float(transaction, "addr2"),
        dist1=_optional_float(transaction, "dist1"),
        dist2=_optional_float(transaction, "dist2"),
        p_emaildomain=_optional_str(transaction, "P_emaildomain"),
        r_emaildomain=_optional_str(transaction, "R_emaildomain"),
        counts=_float_map(transaction, COUNT_COLUMNS),
        deltas=_float_map(transaction, DELTA_COLUMNS),
        match_flags=_string_map(transaction, MATCH_COLUMNS),
        vesta=_float_map(transaction, VESTA_COLUMNS),
        identity_numeric=_float_map(identity, IDENTITY_NUMERIC_COLUMNS),
        identity_categorical=_string_map(identity, IDENTITY_CATEGORICAL_COLUMNS),
        device_type=_optional_str(identity, "DeviceType"),
        device_info=_optional_str(identity, "DeviceInfo"),
    )
