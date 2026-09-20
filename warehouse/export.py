"""Publish the Delta tables to Parquet for the warehouse.

DuckDB has a Delta extension, and it is the obvious thing to reach for. This
project does not use it, for one practical reason: the extension is downloaded
at runtime, so a laptop that is offline - or behind a proxy that blocks the
extension host, which is exactly what happened here - gets a warehouse that
cannot read anything.

An explicit publish step is also honest about what a warehouse is. The Delta
tables are the operational store; the warehouse holds a snapshot that analysts
and dashboards query without contending with the jobs that write them. That
separation is why the Superset/Power BI path can be read-only.

    make export

Writes one Parquet directory per table under ``data/warehouse/export``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from common.config import REPO_ROOT, get_settings
from common.spark import build_spark_session

logger = logging.getLogger("warehouse-export")

DEFAULT_EXPORT_DIR = REPO_ROOT / "data" / "warehouse" / "export"


def exportable_tables(settings) -> dict[str, Path]:
    """The Delta tables the warehouse is built from."""
    return {
        "silver": settings.path(settings.delta_silver_path),
        "gold": settings.path(settings.delta_gold_path),
        "chargebacks": settings.path(settings.delta_chargebacks_path),
        "decisions": settings.path(settings.delta_decisions_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--export-dir", default=str(DEFAULT_EXPORT_DIR))
    parser.add_argument(
        "--tables", nargs="*", help="only export these tables (default: all that exist)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    settings = get_settings()

    tables = exportable_tables(settings)
    wanted = args.tables or list(tables)
    export_dir = Path(args.export_dir)

    spark = build_spark_session("warehouse-export")
    spark.sparkContext.setLogLevel("WARN")

    exported = 0
    for name in wanted:
        source = tables.get(name)
        if source is None:
            logger.warning("unknown table %r - skipping", name)
            continue
        if not source.exists():
            # A table that has not been built yet is not an error: the warehouse
            # is expected to be partial early in a run.
            logger.info("%s does not exist yet - skipping", name)
            continue

        frame = spark.read.format("delta").load(str(source))
        destination = export_dir / name
        # Coalesced: DuckDB reads a handful of files far faster than hundreds
        # of tiny ones, and these snapshots are laptop-sized.
        frame.coalesce(1).write.mode("overwrite").parquet(str(destination))
        logger.info("exported %s (%s rows) -> %s", name, frame.count(), destination)
        exported += 1

    spark.stop()
    logger.info("exported %s tables to %s", exported, export_dir)
    return 0 if exported else 1


if __name__ == "__main__":
    sys.exit(main())
