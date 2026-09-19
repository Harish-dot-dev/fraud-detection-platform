"""Print a summary of the Bronze Delta table.

Quick sanity check after a replay - is the data actually landing, does it cover
the time range you expect, how many cards are in it:

    make bronze-peek
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common.config import get_settings
from common.spark import build_spark_session


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bronze-path", default=str(settings.path(settings.delta_bronze_path)))
    parser.add_argument("--limit", type=int, default=5, help="rows of sample output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not Path(args.bronze_path).exists():
        print(f"No Bronze table at {args.bronze_path}. Run `make produce` first.")
        return 1

    spark = build_spark_session("bronze-inspect", master="local[1]")
    spark.sparkContext.setLogLevel("ERROR")

    from pyspark.sql import functions as F

    bronze = spark.read.format("delta").load(args.bronze_path)
    summary = bronze.select(
        F.count("*").alias("rows"),
        F.countDistinct("transaction_id").alias("distinct_transactions"),
        F.countDistinct("card_key").alias("distinct_cards"),
        F.min("event_time").alias("first_event"),
        F.max("event_time").alias("last_event"),
        F.max("ingested_at").alias("last_ingest"),
    ).first()

    print(f"Bronze table: {args.bronze_path}")
    print(f"  rows                  {summary['rows']:,}")
    print(f"  distinct transactions {summary['distinct_transactions']:,}")
    print(f"  distinct cards        {summary['distinct_cards']:,}")
    print(f"  event time range      {summary['first_event']}  ->  {summary['last_event']}")
    print(f"  last ingested at      {summary['last_ingest']}")

    # Duplicates here would mean the checkpoint is not doing its job.
    if summary["rows"] != summary["distinct_transactions"]:
        print("  WARNING: duplicate transaction_ids present")

    print("\n  rows per day (first 5 partitions):")
    (bronze.groupBy("event_date").count().orderBy("event_date").limit(5).show(truncate=False))

    print(f"  most recent {args.limit} payments:")
    (
        bronze.select("transaction_id", "event_time", "card_key", "amount", "product_cd")
        .orderBy(F.col("event_time").desc())
        .limit(args.limit)
        .show(truncate=False)
    )

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
