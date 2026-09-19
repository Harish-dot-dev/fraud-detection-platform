"""Tests for the Bronze streaming job.

The Kafka source itself is not tested here - that is Spark's code, not ours.
What is tested is the part that can break silently: the schema staying in step
with the Pydantic event model, and the parsing turning a Kafka record into the
Bronze row we expect.

The tests build a DataFrame shaped like Kafka's output rather than reading from
a broker, so they run anywhere Spark runs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from common.events import PaymentEvent
from common.timeline import DEFAULT_EPOCH
from tests.spark_helpers import KAFKA_SCHEMA, kafka_shaped_dataframe

pytestmark = pytest.mark.needs_spark


def test_spark_schema_matches_the_event_model() -> None:
    """The Spark schema and the Pydantic model must describe the same event.

    This is the cheap insurance against the classic failure in a streaming
    project: someone adds a field to the producer, the consumer's explicit
    schema silently drops it, and the column is null for a week before anyone
    notices.
    """
    from streaming.bronze import payment_event_schema

    spark_fields = {field.name for field in payment_event_schema().fields}

    assert spark_fields == set(PaymentEvent.model_fields)


def test_parse_produces_one_row_per_message(spark, sample_transactions: pd.DataFrame) -> None:
    from streaming.bronze import parse_payment_events

    raw = kafka_shaped_dataframe(spark, sample_transactions, n=25)

    parsed = parse_payment_events(raw)

    assert parsed.count() == 25


def test_parsed_columns_carry_the_payment_and_its_provenance(
    spark, sample_transactions: pd.DataFrame
) -> None:
    from streaming.bronze import parse_payment_events

    raw = kafka_shaped_dataframe(spark, sample_transactions, n=5)
    row = parse_payment_events(raw).orderBy("kafka_offset").first()
    expected = sample_transactions.iloc[0]

    # The payment itself...
    assert row["transaction_id"] == int(expected["TransactionID"])
    assert row["amount"] == pytest.approx(float(expected["TransactionAmt"]))
    assert row["card_key"].startswith(str(int(expected["card1"])))
    # ...where it came from...
    assert row["kafka_topic"] == "payments"
    assert row["kafka_offset"] == 0
    assert row["ingested_at"] is not None
    # ...and the untouched original payload, so Bronze can always be replayed.
    assert '"transaction_id":' in row["raw_value"]


def test_event_time_is_parsed_as_a_timestamp(spark, sample_transactions: pd.DataFrame) -> None:
    """The JSON carries an ISO-8601 string; Bronze needs a real timestamp."""
    from streaming.bronze import parse_payment_events

    raw = kafka_shaped_dataframe(spark, sample_transactions, n=5)
    row = parse_payment_events(raw).orderBy("kafka_offset").first()
    expected_dt = int(sample_transactions.iloc[0]["TransactionDT"])

    assert row["event_time"] == DEFAULT_EPOCH.replace(tzinfo=None) + pd.Timedelta(
        seconds=expected_dt
    )
    # The partition column is derived from it.
    assert row["event_date"] == row["event_time"].date()


def test_sparse_blocks_survive_as_maps(spark, sample_transactions: pd.DataFrame) -> None:
    from streaming.bronze import parse_payment_events

    raw = kafka_shaped_dataframe(spark, sample_transactions, n=5)
    row = parse_payment_events(raw).orderBy("kafka_offset").first()

    assert row["counts"]["C1"] is not None
    assert len(row["counts"]) == 14
    assert isinstance(row["vesta"], dict)


def test_unparseable_messages_are_dropped_not_stored_as_nulls(spark) -> None:
    """A malformed message must not become an all-null row in the typed table."""
    from datetime import datetime

    from streaming.bronze import parse_payment_events

    raw = spark.createDataFrame(
        [
            {
                "key": b"broken",
                "value": b"this is not json",
                "topic": "payments",
                "partition": 0,
                "offset": 0,
                "timestamp": datetime(2023, 6, 1),
            }
        ],
        KAFKA_SCHEMA,
    )

    assert parse_payment_events(raw).count() == 0


@pytest.mark.slow
def test_streaming_write_lands_in_a_partitioned_delta_table(
    spark, sample_transactions: pd.DataFrame, tmp_path
) -> None:
    """End-to-end through the real streaming writer, minus Kafka.

    A file source stands in for the Kafka source so that the writer, its
    checkpoint and the partitioning are all genuinely exercised.
    """
    from streaming.bronze import parse_payment_events, write_bronze_stream

    source_dir = tmp_path / "source"
    bronze_path = str(tmp_path / "bronze")
    checkpoint_path = str(tmp_path / "checkpoint")

    kafka_shaped = kafka_shaped_dataframe(spark, sample_transactions, n=40)
    kafka_shaped.write.parquet(str(source_dir))

    stream = spark.readStream.schema(kafka_shaped.schema).parquet(str(source_dir))
    query = write_bronze_stream(
        parse_payment_events(stream), bronze_path, checkpoint_path, once=True
    )
    query.awaitTermination()

    bronze = spark.read.format("delta").load(bronze_path)
    assert bronze.count() == 40
    assert "event_date" in bronze.columns
    # Partitioned by day of the synthetic calendar.
    assert (tmp_path / "bronze").glob("event_date=*")


@pytest.mark.slow
def test_restarting_the_stream_does_not_duplicate_rows(
    spark, sample_transactions: pd.DataFrame, tmp_path
) -> None:
    """The checkpoint is what makes a restart safe.

    Without it, every restart would re-read the topic from the beginning and
    double the table - and the velocity features computed from it.
    """
    from streaming.bronze import parse_payment_events, write_bronze_stream

    source_dir = str(tmp_path / "source")
    bronze_path = str(tmp_path / "bronze")
    checkpoint_path = str(tmp_path / "checkpoint")

    kafka_shaped = kafka_shaped_dataframe(spark, sample_transactions, n=20)
    kafka_shaped.write.parquet(source_dir)

    for _ in range(2):
        stream = spark.readStream.schema(kafka_shaped.schema).parquet(source_dir)
        query = write_bronze_stream(
            parse_payment_events(stream), bronze_path, checkpoint_path, once=True
        )
        query.awaitTermination()

    assert spark.read.format("delta").load(bronze_path).count() == 20
