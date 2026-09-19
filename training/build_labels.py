"""Build the chargeback (label) table.

    make labels

Reads the raw transaction file - the only place in the platform that touches
``isFraud`` - applies the arrival delay from ``training/chargebacks.py``, and
writes a Delta table of labels with the date each one became known.

Airflow reloads this daily in phase 7, which is what makes the delay visible
over time: on any given day only the disputes that have arrived are in hand.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from common.config import get_settings
from common.spark import build_spark_session
from common.timeline import parse_epoch
from producer.replay import resolve_source_paths
from training.chargebacks import build_chargebacks, label_summary

logger = logging.getLogger("labels")


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--transactions", type=Path, help="path to train_transaction.csv")
    parser.add_argument(
        "--chargebacks-path", default=str(settings.path(settings.delta_chargebacks_path))
    )
    parser.add_argument(
        "--fixture", action="store_true", help="use the synthetic fixture even if data/raw exists"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    settings = get_settings()

    if args.transactions:
        transactions_path = args.transactions
    else:
        transactions_path, _, is_real = resolve_source_paths(prefer_raw=not args.fixture)
        if not is_real:
            logger.warning("using the synthetic fixture: %s", transactions_path.name)

    transactions = pd.read_csv(
        transactions_path, usecols=["TransactionID", "isFraud", "TransactionDT"]
    )
    chargebacks = build_chargebacks(
        transactions,
        epoch=parse_epoch(settings.synthetic_epoch),
        seed=settings.random_seed,
        min_delay_days=settings.chargeback_min_delay_days,
        max_delay_days=settings.chargeback_max_delay_days,
        maturity_days=settings.label_maturity_days,
    )

    latest_event = chargebacks["event_time"].max()
    summary = label_summary(chargebacks, latest_event)
    logger.info(
        "%s transactions; on the day of the last payment only %s labels (%.1f%%) would be known",
        summary["transactions"],
        summary["labels_known"],
        summary["known_share"] * 100,
    )

    spark = build_spark_session("build-labels")
    spark.sparkContext.setLogLevel("WARN")
    (
        spark.createDataFrame(chargebacks)
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(args.chargebacks_path)
    )
    logger.info("wrote %s rows to %s", len(chargebacks), args.chargebacks_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
