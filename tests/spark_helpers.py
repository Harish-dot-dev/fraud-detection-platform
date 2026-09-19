"""Helpers for building Spark DataFrames that look like the real pipeline's.

Tests that need a Bronze or Silver table build it by running the production
code over fixture data rather than hand-writing rows: if the Bronze schema
changes, these helpers change with it and the tests keep testing something
real.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from common.events import payment_event_from_row
from common.timeline import DEFAULT_EPOCH

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame, SparkSession

KAFKA_SCHEMA = (
    "key binary, value binary, topic string, partition int, offset long, timestamp timestamp"
)
INGESTED_AT = datetime(2023, 6, 1, 12, 0, 0, tzinfo=UTC)


def kafka_shaped_dataframe(
    spark: SparkSession,
    transactions: pd.DataFrame,
    identity: pd.DataFrame | None = None,
    n: int | None = None,
) -> DataFrame:
    """Build a DataFrame with the columns Spark's Kafka source produces."""
    identity_index: dict[int, dict[str, Any]] = {}
    if identity is not None:
        identity_index = {
            int(row["TransactionID"]): row for row in identity.to_dict(orient="records")
        }

    frame = transactions if n is None else transactions.head(n)
    rows = []
    for offset, record in enumerate(frame.to_dict(orient="records")):
        event = payment_event_from_row(
            record, identity_index.get(int(record["TransactionID"])), epoch=DEFAULT_EPOCH
        )
        rows.append(
            {
                "key": event.card_key.encode(),
                "value": event.model_dump_json().encode(),
                "topic": "payments",
                # Deliberately spread across partitions: a card's events can
                # arrive on different partitions in the tests, which is exactly
                # when ordering bugs show up.
                "partition": offset % 3,
                "offset": offset,
                "timestamp": INGESTED_AT,
            }
        )
    return spark.createDataFrame(rows, KAFKA_SCHEMA)


def bronze_dataframe(
    spark: SparkSession,
    transactions: pd.DataFrame,
    identity: pd.DataFrame | None = None,
    n: int | None = None,
) -> DataFrame:
    """A Bronze-shaped DataFrame, produced by the real parsing code."""
    from streaming.bronze import parse_payment_events

    return parse_payment_events(kafka_shaped_dataframe(spark, transactions, identity, n))


def events_from_fixture(
    transactions: pd.DataFrame, identity: pd.DataFrame | None = None, n: int | None = None
) -> list:
    """The same payments as plain :class:`PaymentEvent` objects, in time order.

    Used as the reference side of the online/offline consistency test.
    """
    identity_index: dict[int, dict[str, Any]] = {}
    if identity is not None:
        identity_index = {
            int(row["TransactionID"]): row for row in identity.to_dict(orient="records")
        }

    frame = transactions if n is None else transactions.head(n)
    frame = frame.sort_values(["TransactionDT", "TransactionID"], kind="stable")
    return [
        payment_event_from_row(
            record, identity_index.get(int(record["TransactionID"])), epoch=DEFAULT_EPOCH
        )
        for record in frame.to_dict(orient="records")
    ]
