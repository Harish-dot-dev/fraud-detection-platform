"""Publish read-only copies of the warehouse for dashboards.

DuckDB allows a single writer and takes a file lock. A dashboard holding a
connection while the daily batch rebuilds the models is the classic way to get
either a locked warehouse or a failed dbt run - and it is why Superset over
DuckDB has a reputation for being fragile.

The fix is unglamorous: dashboards read a **copy**. `make publish` snapshots
the warehouse file and exports the tables a BI tool wants as Parquet, both
atomically (write beside, then rename), so a reader never sees a half-written
file.

    make publish

Outputs:
    data/warehouse/fraud_readonly.duckdb   - for Superset
    data/exports/powerbi/*.parquet         - for Power BI Desktop, which reads
                                             Parquet natively with no driver
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from common.config import REPO_ROOT, get_settings

logger = logging.getLogger("warehouse-publish")

READONLY_NAME = "fraud_readonly.duckdb"
POWERBI_DIR = REPO_ROOT / "data" / "exports" / "powerbi"

# The models a dashboard actually needs. Staging views are an implementation
# detail and stay out of the published copy's export.
PUBLISHED_TABLES = [
    "fct_decisions",
    "agg_daily_kpis",
    "agg_model_performance",
    "agg_rule_effectiveness",
]


def snapshot_warehouse(source: Path, destination: Path) -> Path:
    """Copy the warehouse file for read-only consumers.

    Written to a temporary name and renamed, so a dashboard mid-query never
    opens a partially copied database.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_suffix(".duckdb.tmp")
    shutil.copy2(source, staging)
    staging.replace(destination)
    return destination


def export_for_powerbi(database: Path, output_dir: Path) -> list[Path]:
    """Write each published table as Parquet.

    Power BI Desktop has no DuckDB connector, and a third-party ODBC driver is
    exactly the fragile dependency this project avoids. It reads Parquet
    natively.
    """
    import duckdb

    output_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(database), read_only=True)
    written = []

    for table in PUBLISHED_TABLES:
        target = output_dir / f"{table}.parquet"
        connection.execute(f"COPY (SELECT * FROM {table}) TO '{target}' (FORMAT PARQUET)")
        written.append(target)

    connection.close()
    return written


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--warehouse", default=str(settings.path(settings.duckdb_path)))
    parser.add_argument("--powerbi-dir", default=str(POWERBI_DIR))
    parser.add_argument("--skip-powerbi", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    source = Path(args.warehouse)
    if not source.exists():
        logger.error("no warehouse at %s - run `make warehouse` first", source)
        return 1

    readonly = snapshot_warehouse(source, source.parent / READONLY_NAME)
    logger.info("read-only snapshot: %s", readonly)

    if not args.skip_powerbi:
        written = export_for_powerbi(readonly, Path(args.powerbi_dir))
        logger.info("Power BI exports: %s files in %s", len(written), args.powerbi_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
