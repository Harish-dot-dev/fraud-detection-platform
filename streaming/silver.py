"""Bronze -> Silver: clean, deduplicate and tokenise.

Bronze is whatever arrived. Silver is what the rest of the platform is allowed
to use, which means three things happen here:

1. **Deduplication.** A replayed Kafka offset or a re-run must not produce two
   rows for one payment - every velocity feature would be wrong.
2. **Tokenisation.** The card identity proxy becomes an HMAC token, and the
   columns it was built from are dropped. After this layer nothing downstream
   - no model, no dashboard, no LLM prompt - can link a row back to a card.
3. **Cleaning.** Types, obviously invalid amounts, and the derived date column.

The job rebuilds Silver from Bronze rather than appending to it. That is
slower, but it makes the layer a pure function of Bronze: re-running it after
a bug fix produces the correct table instead of a half-corrected one.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import TYPE_CHECKING

from common.config import get_settings
from common.spark import build_spark_session

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame

logger = logging.getLogger("silver")

# Columns removed because together they *are* the card identity proxy. See
# docs/design_decisions.md: keeping them alongside the token would make the
# token pointless.
IDENTIFYING_COLUMNS = ["card1", "card2", "addr1", "addr2", "card_key"]


def bronze_to_silver(bronze: DataFrame, salt: str) -> DataFrame:
    """Transform Bronze rows into the Silver table."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    from common.pii_spark import tokenise_column

    # Keep the first arrival of each transaction. Ordering by the Kafka
    # coordinates makes the choice deterministic rather than "whichever row the
    # shuffle happened to put first".
    arrival_order = Window.partitionBy("transaction_id").orderBy(
        F.col("kafka_partition").asc(), F.col("kafka_offset").asc()
    )

    deduplicated = (
        bronze.withColumn("_row", F.row_number().over(arrival_order))
        .filter(F.col("_row") == 1)
        .drop("_row")
    )

    return (
        deduplicated
        # A zero or negative amount is not a payment; it is a data problem, and
        # it would poison the per-card averages.
        .filter(F.col("amount") > 0)
        .withColumn("card_token", tokenise_column(F.col("card_key"), salt))
        .withColumn("event_date", F.to_date(F.col("event_time")))
        .select(
            "transaction_id",
            "event_time",
            "event_date",
            "transaction_dt",
            "card_token",
            "amount",
            # Coarse payment attributes: none of these identifies a card on its
            # own, and they are genuinely useful to the model.
            "product_cd",
            "card3",
            "card4",
            "card5",
            "card6",
            "dist1",
            "dist2",
            "p_emaildomain",
            "r_emaildomain",
            "counts",
            "deltas",
            "match_flags",
            "vesta",
            "identity_numeric",
            "identity_categorical",
            "device_type",
            "device_info",
            "ingested_at",
        )
    )


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bronze-path", default=str(settings.path(settings.delta_bronze_path)))
    parser.add_argument("--silver-path", default=str(settings.path(settings.delta_silver_path)))
    parser.add_argument(
        "--skip-quality",
        action="store_true",
        help="write the table even if the data quality suite fails",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    settings = get_settings()

    if settings.uses_default_pii_salt:
        logger.warning("PII_HASH_SALT is the example value - tokens are not secret.")

    spark = build_spark_session("bronze-to-silver")
    spark.sparkContext.setLogLevel("WARN")

    bronze = spark.read.format("delta").load(args.bronze_path)
    silver = bronze_to_silver(bronze, settings.pii_hash_salt)

    from quality.expectations import validate_silver

    result = validate_silver(silver)
    logger.info("data quality: %s", result.describe())
    if not result.success and not args.skip_quality:
        logger.error("Silver failed its quality checks; not writing. %s", result.failures)
        return 1

    (
        silver.write.format("delta")
        .mode("overwrite")
        .partitionBy("event_date")
        .option("overwriteSchema", "true")
        .save(args.silver_path)
    )
    logger.info("wrote %s rows to %s", silver.count(), args.silver_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
