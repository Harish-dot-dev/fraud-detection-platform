"""The past-cases store: Postgres + pgvector.

When an analyst picks up a flagged payment, the most useful thing you can put
in front of them is not a model score - it is *"here are four payments that
looked like this one, and here is what they turned out to be"*. That is what
this table holds, and what the LLM retrieves from.

Only **confirmed** cases go in. A case whose chargeback has not arrived has no
outcome to learn from, and retrieving it would let an unresolved payment vouch
for another unresolved payment.

Indexing is deliberately deferred. IVFFlat is an *approximate* index: it sorts
vectors into lists and, by default, searches only one of them. On a small table
that is actively wrong - a three-row table with a hundred lists returned two
neighbours for a request for three, because the third vector was sitting in a
list the query never probed.

So the index is only created once there are enough rows to justify it
(``ensure_index``), and below that threshold Postgres does an exact sequential
scan, which on a few thousand short vectors is both correct and fast. This also
matches pgvector's own advice: build the index after loading data, with roughly
one list per thousand rows.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from genai.embeddings import EMBEDDING_DIMENSIONS, Embedder

logger = logging.getLogger("genai.case_store")

TABLE = "fraud_cases"

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS {TABLE} (
    transaction_id   BIGINT PRIMARY KEY,
    card_token       TEXT NOT NULL,
    occurred_at      TIMESTAMPTZ NOT NULL,
    amount           DOUBLE PRECISION NOT NULL,
    decision         TEXT NOT NULL,
    -- The confirmed outcome. Nothing without one belongs in this table.
    is_fraud         BOOLEAN NOT NULL,
    confirmed_at     TIMESTAMPTZ,
    -- How the case was resolved: a chargeback, or an analyst's own decision
    -- from the review queue (the feedback loop in phase 9).
    resolution       TEXT NOT NULL DEFAULT 'chargeback',
    description      TEXT NOT NULL,
    facts            JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    embedding        vector({EMBEDDING_DIMENSIONS}) NOT NULL
);

CREATE INDEX IF NOT EXISTS {TABLE}_occurred_at_idx ON {TABLE} (occurred_at);
"""

# Below this many rows an exact scan beats an approximate index, and an
# approximate index actively misleads (see the module docstring).
MIN_ROWS_FOR_INDEX = 1000
ROWS_PER_LIST = 1000


@dataclass
class PastCase:
    """One confirmed case, as stored and as retrieved."""

    transaction_id: int
    card_token: str
    occurred_at: datetime
    amount: float
    decision: str
    is_fraud: bool
    description: str
    resolution: str = "chargeback"
    confirmed_at: datetime | None = None
    facts: dict[str, Any] = field(default_factory=dict)
    # Only set on retrieval: cosine similarity to the query, 1.0 is identical.
    similarity: float | None = None

    @property
    def outcome(self) -> str:
        return "confirmed fraud" if self.is_fraud else "confirmed legitimate"

    def summary_line(self) -> str:
        """One line, as it appears in the prompt and in the analyst app."""
        when = self.occurred_at.date() if self.occurred_at else "unknown date"
        return f"{when}: {self.amount:.2f} {self.decision} -> {self.outcome} ({self.resolution})"


def describe_case(
    amount: float,
    decision: str,
    features: dict[str, float],
    product_cd: str | None = None,
    device_type: str | None = None,
) -> str:
    """The text that gets embedded.

    Deliberately plain sentences rather than a dump of numbers: the embedder
    was trained on prose, and "first payment seen on this card" sits closer in
    vector space to another new-card case than "card_is_new=1.0" does.

    Amounts are bucketed for the same reason - 512.40 and 498.10 are the same
    kind of payment, and an embedder should not have to learn that.
    """
    parts = [f"{_amount_band(amount)} payment, decision {decision}"]

    if product_cd:
        parts.append(f"product {product_cd}")
    if device_type:
        parts.append(f"on {device_type}")

    if features.get("card_is_new"):
        parts.append("first payment seen on this card")
    else:
        count_24h = features.get("card_txn_count_24h", 0)
        if count_24h >= 10:
            parts.append(f"high velocity, {int(count_24h)} payments in 24 hours")
        elif count_24h > 1:
            parts.append(f"{int(count_24h)} payments in 24 hours")

    ratio = features.get("amount_to_card_avg_ratio")
    if ratio and ratio >= 3:
        parts.append(f"about {ratio:.0f} times this card's usual amount")
    if features.get("card_new_device"):
        parts.append("device not seen on this card before")
    if features.get("card_new_email_domain"):
        parts.append("new recipient email domain")
    if features.get("is_night"):
        parts.append("made overnight")

    return "; ".join(parts)


def _amount_band(amount: float) -> str:
    for limit, label in ((50, "small"), (250, "moderate"), (1000, "large"), (5000, "very large")):
        if amount < limit:
            return label
    return "exceptional"


class CaseStore:
    """Read and write confirmed cases in pgvector."""

    def __init__(self, connection: Any, embedder: Embedder) -> None:
        self._connection = connection
        self._embedder = embedder

    def create_schema(self) -> None:
        with self._connection.cursor() as cursor:
            cursor.execute(SCHEMA)
        self._connection.commit()

    def ensure_index(self, min_rows: int = MIN_ROWS_FOR_INDEX) -> bool:
        """Create the IVFFlat index once the table is big enough to need it.

        Returns True if an index was created. Called after a bulk load rather
        than at schema creation: pgvector builds a better index when it can see
        the data, and an approximate index over a handful of rows returns
        fewer neighbours than asked for.
        """
        rows = self.count()
        if rows < min_rows:
            logger.info(
                "%s rows: keeping the exact scan (an IVFFlat index needs %s+ to help)",
                rows,
                min_rows,
            )
            return False

        lists = max(1, rows // ROWS_PER_LIST)
        with self._connection.cursor() as cursor:
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS {TABLE}_embedding_idx ON {TABLE} "
                f"USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})"
            )
        self._connection.commit()
        logger.info("created an IVFFlat index over %s rows with %s lists", rows, lists)
        return True

    def add_cases(self, cases: list[PastCase]) -> int:
        """Insert or update cases, embedding their descriptions in one batch."""
        if not cases:
            return 0

        vectors = self._embedder.embed([case.description for case in cases])
        rows = [
            (
                case.transaction_id,
                case.card_token,
                case.occurred_at,
                case.amount,
                case.decision,
                case.is_fraud,
                case.confirmed_at,
                case.resolution,
                case.description,
                json.dumps(case.facts),
                _to_pgvector(vector),
            )
            for case, vector in zip(cases, vectors, strict=True)
        ]

        with self._connection.cursor() as cursor:
            cursor.executemany(
                f"""
                INSERT INTO {TABLE} (
                    transaction_id, card_token, occurred_at, amount, decision,
                    is_fraud, confirmed_at, resolution, description, facts, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (transaction_id) DO UPDATE SET
                    decision = EXCLUDED.decision,
                    is_fraud = EXCLUDED.is_fraud,
                    confirmed_at = EXCLUDED.confirmed_at,
                    resolution = EXCLUDED.resolution,
                    description = EXCLUDED.description,
                    facts = EXCLUDED.facts,
                    embedding = EXCLUDED.embedding
                """,
                rows,
            )
        self._connection.commit()
        return len(rows)

    def find_similar(
        self, description: str, top_k: int = 5, exclude_transaction_id: int | None = None
    ) -> list[PastCase]:
        """The k most similar confirmed cases.

        A payment never retrieves itself: an identical match would be a
        perfect, useless neighbour, and during evaluation it would quietly
        turn retrieval quality into 100%.
        """
        vector = _to_pgvector(self._embedder.embed([description])[0])

        with self._connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT transaction_id, card_token, occurred_at, amount, decision,
                       is_fraud, confirmed_at, resolution, description, facts,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM {TABLE}
                WHERE %s::bigint IS NULL OR transaction_id <> %s::bigint
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vector, exclude_transaction_id, exclude_transaction_id, vector, top_k),
            )
            rows = cursor.fetchall()

        return [
            PastCase(
                transaction_id=row[0],
                card_token=row[1],
                occurred_at=row[2],
                amount=row[3],
                decision=row[4],
                is_fraud=row[5],
                confirmed_at=row[6],
                resolution=row[7],
                description=row[8],
                facts=row[9] if isinstance(row[9], dict) else json.loads(row[9] or "{}"),
                similarity=float(row[10]),
            )
            for row in rows
        ]

    def count(self) -> int:
        with self._connection.cursor() as cursor:
            cursor.execute(f"SELECT count(*) FROM {TABLE}")
            return int(cursor.fetchone()[0])


def _to_pgvector(vector: np.ndarray) -> str:
    """pgvector's text input format."""
    return "[" + ",".join(f"{value:.6f}" for value in np.asarray(vector).ravel()) + "]"


def connect(host: str, port: int, database: str, user: str, password: str) -> Any:
    """Open a psycopg connection to the case store."""
    import psycopg

    return psycopg.connect(host=host, port=port, dbname=database, user=user, password=password)
