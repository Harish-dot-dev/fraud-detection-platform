"""Analyst decisions, and the feedback loop they feed.

When an analyst resolves a case, two things happen:

1. the decision is recorded in ``analyst_reviews`` - who decided what, when,
   and why - which is the audit trail for the *human* half of the system;
2. the case is written into the pgvector store as a **confirmed** case, so the
   next analyst looking at something similar sees it.

That second step is the loop that makes the assistant improve without
retraining anything. An analyst's confirmation arrives in minutes; a chargeback
takes weeks. Both are real outcomes, and the store records which is which
(``resolution``) so nobody mistakes an analyst's judgement for a settled
dispute.

Reviews live in Postgres rather than DuckDB because the warehouse is rebuilt
from scratch by the daily batch - anything written there would be destroyed on
the next run - and because DuckDB takes a single writer, which an interactive
app should never hold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("analyst_app.reviews")

DEFAULT_TABLE = "analyst_reviews"

# The table name is a constructor argument rather than a module constant, so a
# test can point a store at its own table without monkeypatching globals -
# which is how an earlier version of these tests corrupted its own schema
# string between cases.
SCHEMA_TEMPLATE = """
CREATE TABLE IF NOT EXISTS {table} (
    transaction_id  BIGINT PRIMARY KEY,
    reviewed_at     TIMESTAMPTZ NOT NULL,
    analyst         TEXT NOT NULL,
    verdict         TEXT NOT NULL CHECK (verdict IN ('fraud', 'legitimate')),
    note            TEXT NOT NULL DEFAULT '',
    -- What the platform had decided, kept alongside so agreement between the
    -- model and the analyst can be measured without another join.
    platform_decision TEXT NOT NULL,
    model_version   TEXT NOT NULL DEFAULT ''
);
"""

VERDICT_FRAUD = "fraud"
VERDICT_LEGITIMATE = "legitimate"


@dataclass
class Review:
    """One analyst's resolution of one case."""

    transaction_id: int
    verdict: str
    analyst: str
    platform_decision: str
    note: str = ""
    model_version: str = ""
    reviewed_at: datetime | None = None

    @property
    def is_fraud(self) -> bool:
        return self.verdict == VERDICT_FRAUD

    @property
    def agreed_with_platform(self) -> bool:
        """Did the analyst reach the same conclusion the platform did?"""
        flagged_as_fraud = self.platform_decision == "block"
        return flagged_as_fraud == self.is_fraud


class ReviewStore:
    """Records analyst decisions and feeds them back as confirmed cases."""

    def __init__(
        self, connection: Any, case_store: Any | None = None, table: str = DEFAULT_TABLE
    ) -> None:
        self._connection = connection
        self._case_store = case_store
        self.table = table

    def create_schema(self) -> None:
        with self._connection.cursor() as cursor:
            cursor.execute(SCHEMA_TEMPLATE.format(table=self.table))
        self._connection.commit()

    def record(self, review: Review, case_facts: dict[str, Any] | None = None) -> None:
        """Save a review, and add it to the case store as a confirmed case."""
        reviewed_at = review.reviewed_at or datetime.now(UTC)

        with self._connection.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {self.table} (
                    transaction_id, reviewed_at, analyst, verdict, note,
                    platform_decision, model_version
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (transaction_id) DO UPDATE SET
                    reviewed_at = EXCLUDED.reviewed_at,
                    analyst = EXCLUDED.analyst,
                    verdict = EXCLUDED.verdict,
                    note = EXCLUDED.note
                """,
                (
                    review.transaction_id,
                    reviewed_at,
                    review.analyst,
                    review.verdict,
                    review.note,
                    review.platform_decision,
                    review.model_version,
                ),
            )
        self._connection.commit()

        if self._case_store is not None and case_facts:
            self._add_to_case_store(review, reviewed_at, case_facts)

    def _add_to_case_store(
        self, review: Review, reviewed_at: datetime, case_facts: dict[str, Any]
    ) -> None:
        """Close the loop: this case is now available to retrieval."""
        from genai.case_store import PastCase, describe_case

        features = case_facts.get("features", {})
        amount = float(features.get("amount", case_facts.get("amount", 0.0)))

        try:
            self._case_store.add_cases(
                [
                    PastCase(
                        transaction_id=review.transaction_id,
                        card_token=case_facts.get("card_token", ""),
                        occurred_at=case_facts.get("decided_at") or reviewed_at,
                        amount=amount,
                        decision=review.platform_decision,
                        is_fraud=review.is_fraud,
                        confirmed_at=reviewed_at,
                        # Not a chargeback: an analyst's judgement, minutes
                        # after the fact rather than weeks.
                        resolution="analyst_review",
                        description=describe_case(
                            amount=amount,
                            decision=review.platform_decision,
                            features=features,
                            product_cd=case_facts.get("product_cd"),
                            device_type=case_facts.get("device_type"),
                        ),
                        facts={"features": features, "note": review.note},
                    )
                ]
            )
        except Exception as error:  # noqa: BLE001 - the review itself is saved
            logger.warning("could not add case %s to the store: %s", review.transaction_id, error)

    def reviewed_ids(self) -> set[int]:
        """Cases already resolved, so the queue does not show them again."""
        with self._connection.cursor() as cursor:
            cursor.execute(f"SELECT transaction_id FROM {self.table}")
            return {int(row[0]) for row in cursor.fetchall()}

    def recent(self, limit: int = 50) -> list[Review]:
        with self._connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT transaction_id, verdict, analyst, platform_decision, note,
                       model_version, reviewed_at
                FROM {self.table} ORDER BY reviewed_at DESC LIMIT %s
                """,
                (limit,),
            )
            return [
                Review(
                    transaction_id=row[0],
                    verdict=row[1],
                    analyst=row[2],
                    platform_decision=row[3],
                    note=row[4],
                    model_version=row[5],
                    reviewed_at=row[6],
                )
                for row in cursor.fetchall()
            ]

    def agreement_rate(self) -> float | None:
        """How often analysts agree with the platform.

        A number worth watching in both directions: near 100% may mean the
        queue is too easy to be worth a human's time; very low means the
        thresholds are wrong.
        """
        reviews = self.recent(limit=1000)
        if not reviews:
            return None
        return sum(review.agreed_with_platform for review in reviews) / len(reviews)
