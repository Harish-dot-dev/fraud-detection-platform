"""Build a training-set-shaped frame without Spark.

The Gold table is produced by Spark, but a model does not care where its rows
came from - so the model tests build the same shape by folding the fixture
events through the shared feature definitions. That keeps the training tests in
the fast suite, where they get run.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from common.events import (
    COUNT_COLUMNS,
    DELTA_COLUMNS,
    PaymentEvent,
    payment_event_from_row,
)
from common.timeline import DEFAULT_EPOCH
from features.definitions import CardState, compute_features, update_state
from tests.synthetic import generate_transactions
from training.chargebacks import build_chargebacks

PASSTHROUGH = {
    "product_cd": "ProductCD",
    "card3": "card3",
    "card4": "card4",
    "card5": "card5",
    "card6": "card6",
    "dist1": "dist1",
    "dist2": "dist2",
    "p_emaildomain": "P_emaildomain",
    "r_emaildomain": "R_emaildomain",
}


def _map_of(row: dict[str, Any], columns: list[str]) -> dict[str, float]:
    return {
        column: float(row[column])
        for column in columns
        if row.get(column) is not None and row.get(column) == row.get(column)
    }


def synthetic_training_set(n_rows: int = 4000, seed: int = 11) -> pd.DataFrame:
    """A labelled, point-in-time-correct training frame.

    Features are folded one payment at a time, so - exactly as in production -
    a row can only see the payments that preceded it on the same card.
    """
    transactions, identity = generate_transactions(n_rows=n_rows, seed=seed)
    identity_index = {int(row["TransactionID"]): row for row in identity.to_dict(orient="records")}
    labels = build_chargebacks(transactions).set_index("transaction_id")

    states: dict[str, CardState] = {}
    rows: list[dict[str, Any]] = []

    for record in transactions.to_dict(orient="records"):
        transaction_id = int(record["TransactionID"])
        identity_row = identity_index.get(transaction_id)
        event: PaymentEvent = payment_event_from_row(record, identity_row, epoch=DEFAULT_EPOCH)

        state = states.get(event.card_key, CardState())
        features = compute_features(state, event)
        states[event.card_key] = update_state(state, event)

        rows.append(
            {
                "transaction_id": transaction_id,
                "event_time": event.event_time.replace(tzinfo=None),
                "card_token": event.card_key,
                **features,
                **{target: record.get(source) for target, source in PASSTHROUGH.items()},
                "device_type": (identity_row or {}).get("DeviceType"),
                "counts": _map_of(record, COUNT_COLUMNS),
                "deltas": _map_of(record, DELTA_COLUMNS),
                "is_fraud": int(record["isFraud"]),
                "label_available_at": labels.loc[transaction_id, "label_available_at"],
                "label_is_assumed": False,
            }
        )

    return pd.DataFrame(rows)
