"""Populate the case store from confirmed historical decisions.

    make load-cases

Only payments whose label has actually arrived go in. A case with no confirmed
outcome has nothing to teach: retrieving it would let one unresolved payment
vouch for another, which is how a retrieval system starts confidently
reinforcing its own mistakes.

The descriptions are written by ``describe_case`` rather than dumped as
numbers, because the embedder was trained on prose - "first payment seen on
this card" lands near another new-card case in vector space in a way that
``card_is_new=1.0`` does not.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from common.config import REPO_ROOT, get_settings
from genai.case_store import CaseStore, PastCase, connect, describe_case
from genai.embeddings import build_embedder

logger = logging.getLogger("load-cases")

DEFAULT_EXPORT_DIR = REPO_ROOT / "data" / "warehouse" / "export"


def load_confirmed_cases(export_dir: Path, limit: int = 0) -> list[PastCase]:
    """Read decisions whose labels have matured, as cases."""
    import duckdb

    connection = duckdb.connect()
    query = f"""
        WITH latest AS (
            SELECT *, row_number() OVER (
                PARTITION BY transaction_id ORDER BY decided_at DESC
            ) AS rank
            FROM read_parquet('{export_dir}/decisions/*.parquet')
        )
        SELECT
            d.transaction_id, d.card_token, d.decided_at, d.decision,
            d.features, c.is_fraud, c.label_available_at, c.label_source,
            s.product_cd, s.device_type
        FROM latest d
        JOIN read_parquet('{export_dir}/chargebacks/*.parquet') c
          USING (transaction_id)
        LEFT JOIN read_parquet('{export_dir}/silver/*.parquet') s
          USING (transaction_id)
        WHERE d.rank = 1
          -- Only confirmed outcomes.
          AND c.label_available_at <= current_timestamp
        ORDER BY d.decided_at
        {f"LIMIT {limit}" if limit else ""}
    """
    rows = connection.execute(query).fetchall()

    cases = []
    for row in rows:
        features = dict(row[4] or {})
        amount = float(features.get("amount", 0.0))
        cases.append(
            PastCase(
                transaction_id=int(row[0]),
                card_token=row[1],
                occurred_at=row[2],
                amount=amount,
                decision=row[3],
                is_fraud=bool(row[5]),
                confirmed_at=row[6],
                resolution=row[7] or "chargeback",
                description=describe_case(
                    amount=amount,
                    decision=row[3],
                    features=features,
                    product_cd=row[8],
                    device_type=row[9],
                ),
                facts={"features": features},
            )
        )
    return cases


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--export-dir", default=str(DEFAULT_EXPORT_DIR))
    parser.add_argument("--limit", type=int, default=0, help="most recent N cases (0 = all)")
    parser.add_argument("--host", default=settings.pgvector_host)
    parser.add_argument("--port", type=int, default=settings.pgvector_port)
    parser.add_argument("--database", default=settings.pgvector_db)
    parser.add_argument("--user", default=settings.pgvector_user)
    parser.add_argument("--password", default=settings.pgvector_password)
    parser.add_argument(
        "--allow-stub-embedder",
        action="store_true",
        help="fall back to the deterministic stub if the real model is unavailable",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    settings = get_settings()

    cases = load_confirmed_cases(Path(args.export_dir), args.limit)
    if not cases:
        logger.error("no confirmed cases in %s - run `make export` first", args.export_dir)
        return 1

    embedder = build_embedder(settings, allow_stub=args.allow_stub_embedder)
    connection = connect(args.host, args.port, args.database, args.user, args.password)
    store = CaseStore(connection, embedder)
    store.create_schema()

    written = store.add_cases(cases)
    store.ensure_index()

    fraud = sum(case.is_fraud for case in cases)
    logger.info(
        "loaded %s confirmed cases (%s fraud, %s legitimate) with %s",
        written,
        fraud,
        written - fraud,
        embedder.name,
    )
    connection.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
