"""Silver -> Gold: the offline feature table.

Gold is what training sets are built from (phase 4). One row per payment, with
the same feature values the scoring API would have computed for it at the time
- which is what makes a model trained on this table safe to deploy against the
online path.

Like Silver, this job rebuilds the table rather than appending, so it is a pure
function of its input.
"""

from __future__ import annotations

import argparse
import logging
import sys

from common.config import get_settings
from common.spark import build_spark_session
from features.offline import build_gold

logger = logging.getLogger("gold")


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--silver-path", default=str(settings.path(settings.delta_silver_path)))
    parser.add_argument("--gold-path", default=str(settings.path(settings.delta_gold_path)))
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    spark = build_spark_session("silver-to-gold")
    spark.sparkContext.setLogLevel("WARN")

    silver = spark.read.format("delta").load(args.silver_path)
    gold = build_gold(silver)

    (
        gold.write.format("delta")
        .mode("overwrite")
        .partitionBy("event_date")
        .option("overwriteSchema", "true")
        .save(args.gold_path)
    )
    logger.info("wrote %s rows to %s", gold.count(), args.gold_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
