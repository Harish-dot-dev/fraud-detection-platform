"""Simulated chargeback arrival: labels that show up weeks after the payment.

In a real fraud system you do not find out that a payment was fraudulent when
it happens. You find out when the cardholder notices and disputes it, and the
chargeback works its way through the scheme - typically somewhere between a
week and two months later. Legitimate payments are never confirmed at all;
they are simply *presumed* good once the dispute window has closed.

That delay is the single most important operational fact about fraud modelling
and the one most often ignored in tutorials, which treat the label as available
the instant the transaction lands. It is not, and pretending otherwise inflates
every offline metric you will ever report.

The IEEE-CIS dataset gives us ``isFraud`` immediately, so this module puts the
delay back:

* **fraud** -> a chargeback arrives ``CHARGEBACK_MIN_DELAY_DAYS`` to
  ``CHARGEBACK_MAX_DELAY_DAYS`` after the payment. The label is known then.
* **non-fraud** -> nothing arrives. The payment is presumed legitimate once
  ``LABEL_MATURITY_DAYS`` have passed with no dispute.

The delay for a given payment is derived from its transaction ID rather than
drawn from a running random number generator, so it is identical whether the
table is built in one pass or loaded day by day by Airflow. A label timeline
that changed when you rebuilt it would make every point-in-time guarantee
meaningless.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta

import pandas as pd

from common.timeline import DEFAULT_EPOCH, to_event_time

logger = logging.getLogger("chargebacks")

LABEL_SOURCE_CHARGEBACK = "chargeback"
LABEL_SOURCE_MATURED = "matured"

CHARGEBACK_COLUMNS = [
    "transaction_id",
    "is_fraud",
    "event_time",
    "label_available_at",
    "label_source",
    "delay_days",
]


def _deterministic_fraction(transaction_id: int, seed: int) -> float:
    """A stable pseudo-random number in [0, 1) for one transaction.

    Hashing (transaction_id, seed) rather than drawing from a generator means
    the value does not depend on how many rows were processed before it. Build
    the table in one pass or in sixty daily chunks: same answer.
    """
    digest = hashlib.sha256(f"{seed}:{transaction_id}".encode()).digest()
    # The first eight bytes give plenty of resolution for a delay in days.
    return int.from_bytes(digest[:8], "big") / 2**64


def chargeback_delay_days(
    transaction_id: int,
    is_fraud: bool,
    seed: int = 42,
    min_delay_days: int = 7,
    max_delay_days: int = 60,
    maturity_days: int = 60,
) -> float:
    """How long after the payment its label becomes known.

    Fraud gets a delay drawn from the dispute window; everything else is only
    confirmed once the maturity period has elapsed.
    """
    if not is_fraud:
        return float(maturity_days)

    span = max(max_delay_days - min_delay_days, 0)
    return float(min_delay_days + _deterministic_fraction(transaction_id, seed) * span)


def build_chargebacks(
    transactions: pd.DataFrame,
    epoch: datetime = DEFAULT_EPOCH,
    seed: int = 42,
    min_delay_days: int = 7,
    max_delay_days: int = 60,
    maturity_days: int = 60,
) -> pd.DataFrame:
    """Build the label table from the raw transaction file.

    Args:
        transactions: rows of ``train_transaction.csv`` - the only place in the
            platform where ``isFraud`` is read.
        epoch: reference date for the synthetic timeline.

    Returns:
        One row per transaction with the label and *when it became known*.

    Note what this table does **not** feed: nothing on the streaming or scoring
    path reads it. Labels enter the platform here and are used only to build
    training sets and to measure performance after the fact.
    """
    required = {"TransactionID", "isFraud", "TransactionDT"}
    missing = required - set(transactions.columns)
    if missing:
        raise ValueError(f"chargeback simulation needs {sorted(missing)} in the source data")

    frame = transactions[["TransactionID", "isFraud", "TransactionDT"]].copy()
    frame = frame.rename(columns={"TransactionID": "transaction_id", "isFraud": "is_fraud"})
    frame["is_fraud"] = frame["is_fraud"].astype(int)
    frame["event_time"] = frame["TransactionDT"].map(lambda dt: to_event_time(dt, epoch))

    frame["delay_days"] = [
        chargeback_delay_days(
            transaction_id=int(row.transaction_id),
            is_fraud=bool(row.is_fraud),
            seed=seed,
            min_delay_days=min_delay_days,
            max_delay_days=max_delay_days,
            maturity_days=maturity_days,
        )
        for row in frame.itertuples()
    ]
    frame["label_available_at"] = frame["event_time"] + frame["delay_days"].map(
        lambda days: timedelta(days=days)
    )
    frame["label_source"] = frame["is_fraud"].map(
        lambda fraud: LABEL_SOURCE_CHARGEBACK if fraud else LABEL_SOURCE_MATURED
    )

    return frame[CHARGEBACK_COLUMNS].sort_values("label_available_at", kind="stable")


def available_as_of(chargebacks: pd.DataFrame, as_of: datetime) -> pd.DataFrame:
    """The labels a system would actually have in hand on ``as_of``.

    This is what Airflow's daily load simulates: on any given day, only the
    disputes that have already arrived are known.
    """
    return chargebacks[chargebacks["label_available_at"] <= as_of]


def label_summary(chargebacks: pd.DataFrame, as_of: datetime) -> dict[str, float]:
    """A quick picture of how much of the data is usable on a given date."""
    known = available_as_of(chargebacks, as_of)
    total = len(chargebacks)
    return {
        "transactions": total,
        "labels_known": len(known),
        "labels_pending": total - len(known),
        "known_share": len(known) / total if total else 0.0,
        "fraud_rate_known": float(known["is_fraud"].mean()) if len(known) else 0.0,
        "fraud_rate_overall": float(chargebacks["is_fraud"].mean()) if total else 0.0,
    }
