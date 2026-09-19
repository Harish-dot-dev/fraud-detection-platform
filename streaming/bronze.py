"""Bronze layer: land raw payment events from Kafka into Delta Lake.

The Bronze table is deliberately dumb. It parses the JSON into typed columns so
the table is queryable, but it also keeps the original payload string and the
Kafka coordinates of every message. If a downstream assumption turns out to be
wrong, Bronze can be replayed; if the raw payload were dropped, it could not.

    make stream           # continuous, 10-second micro-batches
    make stream-once      # process whatever is in the topic, then exit

Exactly-once-ish semantics come from the checkpoint directory: Spark records
the Kafka offsets it has committed alongside the Delta transaction, so
restarting the job resumes where it stopped rather than duplicating rows.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import TYPE_CHECKING

from common.config import get_settings
from common.spark import KAFKA_PACKAGE, build_spark_session

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame, SparkSession
    from pyspark.sql.streaming import StreamingQuery

logger = logging.getLogger("bronze")

DEFAULT_TRIGGER_INTERVAL = "10 seconds"


def payment_event_schema():
    """Spark schema mirroring :class:`common.events.PaymentEvent`.

    Kept in step with the Pydantic model by ``tests/test_bronze.py``, which
    compares the two field lists. Declaring it explicitly (rather than letting
    Spark infer it) matters for a stream: inference would look at the first
    micro-batch only, and a later batch with a missing optional field would
    change the schema underneath the table.
    """
    from pyspark.sql.types import (
        DoubleType,
        LongType,
        MapType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    float_map = MapType(StringType(), DoubleType())
    string_map = MapType(StringType(), StringType())

    return StructType(
        [
            StructField("transaction_id", LongType(), nullable=False),
            StructField("event_time", TimestampType(), nullable=False),
            StructField("transaction_dt", LongType(), nullable=False),
            StructField("card_key", StringType(), nullable=False),
            StructField("amount", DoubleType(), nullable=False),
            StructField("product_cd", StringType()),
            StructField("card1", LongType()),
            StructField("card2", DoubleType()),
            StructField("card3", DoubleType()),
            StructField("card4", StringType()),
            StructField("card5", DoubleType()),
            StructField("card6", StringType()),
            StructField("addr1", DoubleType()),
            StructField("addr2", DoubleType()),
            StructField("dist1", DoubleType()),
            StructField("dist2", DoubleType()),
            StructField("p_emaildomain", StringType()),
            StructField("r_emaildomain", StringType()),
            StructField("counts", float_map),
            StructField("deltas", float_map),
            StructField("match_flags", string_map),
            StructField("vesta", float_map),
            StructField("identity_numeric", float_map),
            StructField("identity_categorical", string_map),
            StructField("device_type", StringType()),
            StructField("device_info", StringType()),
        ]
    )


def read_payments_stream(
    spark: SparkSession,
    bootstrap_servers: str,
    topic: str,
    starting_offsets: str = "earliest",
    max_offsets_per_trigger: int | None = 20_000,
) -> DataFrame:
    """Open the Kafka source for the payments topic.

    ``maxOffsetsPerTrigger`` bounds how much a single micro-batch reads. Without
    it, the first batch after a long replay tries to swallow the entire topic
    and the job dies on a laptop-sized heap.
    """
    reader = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("subscribe", topic)
        .option("startingOffsets", starting_offsets)
        # A restart must not fail just because the topic was compacted or
        # recreated; Bronze is replayable, so carrying on is the right call.
        .option("failOnDataLoss", "false")
    )
    if max_offsets_per_trigger:
        reader = reader.option("maxOffsetsPerTrigger", str(max_offsets_per_trigger))
    return reader.load()


def parse_payment_events(raw: DataFrame) -> DataFrame:
    """Turn Kafka records into the Bronze table shape.

    Works on both a streaming and a batch DataFrame, which is what lets the
    tests exercise it with a handful of rows and no broker.
    """
    from pyspark.sql import functions as F

    parsed = raw.select(
        F.col("key").cast("string").alias("kafka_key"),
        F.col("value").cast("string").alias("raw_value"),
        F.col("topic").alias("kafka_topic"),
        F.col("partition").alias("kafka_partition"),
        F.col("offset").alias("kafka_offset"),
        F.col("timestamp").alias("kafka_timestamp"),
        F.from_json(F.col("value").cast("string"), payment_event_schema()).alias("event"),
    )

    return (
        parsed
        # A message that does not parse is kept out of the typed table rather
        # than poisoning it with an all-null row. Counting these is a phase 7
        # data-quality check.
        .filter(F.col("event.transaction_id").isNotNull()).select(
            "event.*",
            "kafka_key",
            "raw_value",
            "kafka_topic",
            "kafka_partition",
            "kafka_offset",
            "kafka_timestamp",
            # When this row landed, as opposed to when the payment happened.
            F.current_timestamp().alias("ingested_at"),
            # Partition column: one directory per day of the synthetic calendar.
            F.to_date(F.col("event.event_time")).alias("event_date"),
        )
    )


def write_bronze_batch(events: DataFrame, path: str) -> None:
    """Append one micro-batch to the Bronze table.

    Used by the combined streaming job in ``streaming/features.py``, which
    writes Bronze and updates the online feature store from the same batch
    rather than reading Kafka twice.
    """
    events.write.format("delta").mode("append").partitionBy("event_date").save(path)


def write_bronze_stream(
    events: DataFrame,
    path: str,
    checkpoint_path: str,
    once: bool = False,
    trigger_interval: str = DEFAULT_TRIGGER_INTERVAL,
) -> StreamingQuery:
    """Append parsed events to the Bronze Delta table."""
    writer = (
        events.writeStream.format("delta")
        .outputMode("append")
        .partitionBy("event_date")
        .option("checkpointLocation", checkpoint_path)
    )
    # availableNow drains whatever is in the topic and stops - ideal for tests,
    # demos and Airflow-style batch catch-up.
    writer = (
        writer.trigger(availableNow=True)
        if once
        else writer.trigger(processingTime=trigger_interval)
    )
    return writer.start(path)


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bootstrap-servers", default=settings.kafka_bootstrap_servers)
    parser.add_argument("--topic", default=settings.kafka_topic_payments)
    parser.add_argument("--bronze-path", default=str(settings.path(settings.delta_bronze_path)))
    parser.add_argument(
        "--checkpoint-path",
        default=str(settings.path(settings.delta_bronze_path) / "_checkpoints" / "bronze"),
    )
    parser.add_argument("--starting-offsets", default="earliest", choices=["earliest", "latest"])
    parser.add_argument(
        "--once",
        action="store_true",
        help="process the available data and exit instead of running continuously",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    spark = build_spark_session("bronze-ingest", packages=[KAFKA_PACKAGE])
    spark.sparkContext.setLogLevel("WARN")

    raw = read_payments_stream(
        spark, args.bootstrap_servers, args.topic, starting_offsets=args.starting_offsets
    )
    events = parse_payment_events(raw)

    logger.info("writing bronze to %s (checkpoint %s)", args.bronze_path, args.checkpoint_path)
    query = write_bronze_stream(events, args.bronze_path, args.checkpoint_path, once=args.once)
    query.awaitTermination()

    logger.info("stream finished: %s", query.lastProgress)
    return 0


if __name__ == "__main__":
    sys.exit(main())
