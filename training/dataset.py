"""Point-in-time correct training sets.

A training row may only contain information that existed when the payment was
made, and may only carry a label that had actually arrived by the time the
model was trained. Break either rule and the offline metrics look wonderful
and production does not.

The features half is handled upstream: everything in the Gold table is computed
from a window ending at the payment's own timestamp (``features/offline.py``).
This module handles the label half, which is where the delay lives:

    as_of = the day the model is being trained

    include a payment when   event_time         <= as_of   (it has happened)
                       and   label_available_at <= as_of   (its label has arrived)

The second condition is the one tutorials skip. On any given day, the most
recent weeks of payments have no usable label yet - the disputes simply have
not come in - so a model trained today cannot learn from them. Pretending
otherwise means training on labels from the future.

Two policies for those immature payments are supported, because real teams
disagree about it:

* ``exclude`` (default) - leave them out. Honest, and it throws data away.
* ``assume_legitimate`` - include them as non-fraud, which is what a system
  that trusts "no dispute yet" effectively does. It uses more data and biases
  the model towards calling recent fraud legitimate. The rows are flagged so
  the choice is visible rather than baked in.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from typing import TYPE_CHECKING

from common.config import get_settings
from common.spark import build_spark_session
from common.timeline import parse_epoch

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame

logger = logging.getLogger("dataset")

IMMATURE_EXCLUDE = "exclude"
IMMATURE_ASSUME_LEGITIMATE = "assume_legitimate"
IMMATURE_POLICIES = (IMMATURE_EXCLUDE, IMMATURE_ASSUME_LEGITIMATE)


def build_training_set(
    gold: DataFrame,
    labels: DataFrame,
    as_of: datetime,
    immature_policy: str = IMMATURE_EXCLUDE,
) -> DataFrame:
    """Join features to the labels that were known on ``as_of``.

    Args:
        gold: the offline feature table, one row per payment.
        labels: the chargeback table (``training/chargebacks.py``).
        as_of: the moment the training set is being assembled.
        immature_policy: what to do with payments whose label has not arrived.

    Returns:
        Features plus ``is_fraud``, with ``label_is_assumed`` marking rows whose
        label was inferred rather than observed.
    """
    if immature_policy not in IMMATURE_POLICIES:
        raise ValueError(f"immature_policy must be one of {IMMATURE_POLICIES}")

    from pyspark.sql import functions as F

    as_of_column = F.lit(as_of).cast("timestamp")

    # A payment that has not happened yet cannot be trained on. Obvious, and
    # cheap to enforce - a wrong as_of would otherwise silently pull in the
    # future.
    happened = gold.filter(F.col("event_time") <= as_of_column)

    joined = happened.join(
        labels.select("transaction_id", "is_fraud", "label_available_at", "label_source"),
        on="transaction_id",
        how="inner",
    )

    label_has_arrived = F.col("label_available_at") <= as_of_column

    if immature_policy == IMMATURE_EXCLUDE:
        return (
            joined.filter(label_has_arrived)
            .withColumn("label_is_assumed", F.lit(False))
            .withColumn("as_of", as_of_column)
        )

    return (
        joined.withColumn(
            "is_fraud", F.when(label_has_arrived, F.col("is_fraud")).otherwise(F.lit(0))
        )
        .withColumn("label_is_assumed", ~label_has_arrived)
        .withColumn("as_of", as_of_column)
    )


def describe_training_set(training_set: DataFrame) -> dict[str, float]:
    """Summary numbers worth logging every time a dataset is built."""
    from pyspark.sql import functions as F

    row = training_set.agg(
        F.count("*").alias("rows"),
        F.sum("is_fraud").alias("fraud"),
        F.min("event_time").alias("first_event"),
        F.max("event_time").alias("last_event"),
        F.sum(F.col("label_is_assumed").cast("int")).alias("assumed_labels"),
    ).first()

    rows = int(row["rows"] or 0)
    fraud = int(row["fraud"] or 0)
    return {
        "rows": rows,
        "fraud": fraud,
        "fraud_rate": fraud / rows if rows else 0.0,
        "assumed_labels": int(row["assumed_labels"] or 0),
        "first_event": row["first_event"],
        "last_event": row["last_event"],
    }


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gold-path", default=str(settings.path(settings.delta_gold_path)))
    parser.add_argument(
        "--chargebacks-path", default=str(settings.path(settings.delta_chargebacks_path))
    )
    parser.add_argument("--training-path", default=str(settings.path(settings.delta_training_path)))
    parser.add_argument(
        "--as-of",
        help=(
            "ISO date the training set is assembled on. Defaults to the latest "
            "payment in Gold, i.e. 'now' in the replayed timeline."
        ),
    )
    parser.add_argument(
        "--immature-policy", default=IMMATURE_EXCLUDE, choices=list(IMMATURE_POLICIES)
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    from pyspark.sql import functions as F

    spark = build_spark_session("training-set")
    spark.sparkContext.setLogLevel("WARN")

    gold = spark.read.format("delta").load(args.gold_path)
    labels = spark.read.format("delta").load(args.chargebacks_path)

    if args.as_of:
        as_of = parse_epoch(args.as_of)
    else:
        as_of = gold.agg(F.max("event_time")).first()[0]
        logger.info("no --as-of given; using the latest payment in Gold: %s", as_of)

    training_set = build_training_set(gold, labels, as_of, args.immature_policy)
    summary = describe_training_set(training_set)

    logger.info(
        "as of %s: %s rows, %s fraud (%.3f%%), %s assumed labels, covering %s to %s",
        as_of,
        summary["rows"],
        summary["fraud"],
        summary["fraud_rate"] * 100,
        summary["assumed_labels"],
        summary["first_event"],
        summary["last_event"],
    )
    if summary["rows"] == 0:
        logger.error("empty training set - is as_of earlier than the first matured label?")
        return 1

    (
        training_set.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(args.training_path)
    )
    logger.info("wrote %s", args.training_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
