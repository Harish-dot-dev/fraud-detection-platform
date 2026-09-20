"""Land the decisions topic in Delta.

The scoring API publishes every decision to Kafka and returns; this job is what
makes those decisions queryable. Keeping it separate is the point - analytics
must never be able to slow down or break a payment.

    make decisions-sink          # continuous
    make decisions-sink-once     # drain and exit

The table it writes is the backbone of the dashboards: alert volumes, false
positive rates, model performance over time and scoring latency all come from
here, joined to the chargeback labels once those arrive.
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

logger = logging.getLogger("decisions-sink")


def decision_schema():
    """Spark schema mirroring :class:`serving.decisions.Decision`."""
    from pyspark.sql.types import (
        ArrayType,
        BooleanType,
        DoubleType,
        LongType,
        MapType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    reason = StructType(
        [
            StructField("feature", StringType()),
            StructField("value", StringType()),
            StructField("contribution", DoubleType()),
            StructField("direction", StringType()),
        ]
    )

    return StructType(
        [
            StructField("transaction_id", LongType(), nullable=False),
            StructField("card_token", StringType(), nullable=False),
            StructField("decided_at", TimestampType(), nullable=False),
            StructField("decision", StringType(), nullable=False),
            StructField("score", DoubleType()),
            StructField("triggered_by", StringType()),
            StructField("reason", StringType()),
            StructField("rule_name", StringType()),
            StructField("matched_rules", ArrayType(StringType())),
            StructField("model_version", StringType()),
            StructField("review_threshold", DoubleType()),
            StructField("block_threshold", DoubleType()),
            StructField("top_reasons", ArrayType(reason)),
            StructField("features", MapType(StringType(), DoubleType())),
            StructField("latency_ms", DoubleType()),
            StructField("degraded", BooleanType()),
        ]
    )


def parse_decisions(raw: DataFrame) -> DataFrame:
    """Turn Kafka records into the decisions table shape."""
    from pyspark.sql import functions as F

    return (
        raw.select(
            F.from_json(F.col("value").cast("string"), decision_schema()).alias("decision"),
            F.col("timestamp").alias("kafka_timestamp"),
        )
        .filter(F.col("decision.transaction_id").isNotNull())
        .select(
            "decision.*",
            "kafka_timestamp",
            F.to_date(F.col("decision.decided_at")).alias("decision_date"),
        )
    )


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bootstrap-servers", default=settings.kafka_bootstrap_servers)
    parser.add_argument("--topic", default=settings.kafka_topic_decisions)
    parser.add_argument(
        "--decisions-path", default=str(settings.path(settings.delta_decisions_path))
    )
    parser.add_argument(
        "--checkpoint-path",
        default=str(settings.path(settings.delta_decisions_path) / "_checkpoints" / "sink"),
    )
    parser.add_argument("--once", action="store_true", help="drain the topic and exit")
    parser.add_argument("--trigger-interval", default="30 seconds")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    spark: SparkSession = build_spark_session("decisions-sink", packages=[KAFKA_PACKAGE])
    spark.sparkContext.setLogLevel("WARN")

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", args.bootstrap_servers)
        .option("subscribe", args.topic)
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")
        .load()
    )

    writer = (
        parse_decisions(raw)
        .writeStream.format("delta")
        .outputMode("append")
        .partitionBy("decision_date")
        .option("checkpointLocation", args.checkpoint_path)
    )
    writer = (
        writer.trigger(availableNow=True)
        if args.once
        else writer.trigger(processingTime=args.trigger_interval)
    )

    logger.info("decisions -> %s", args.decisions_path)
    query = writer.start(args.decisions_path)
    query.awaitTermination()
    return 0


if __name__ == "__main__":
    sys.exit(main())
