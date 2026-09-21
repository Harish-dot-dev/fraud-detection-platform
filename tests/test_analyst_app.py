"""Tests for the analyst app's data and feedback loop.

The Streamlit page itself is verified by rendering it in a browser (see
PROGRESS.md); these cover the parts underneath it, where a silent failure would
cost an analyst's decision.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from analyst_app.reviews import (
    VERDICT_FRAUD,
    VERDICT_LEGITIMATE,
    Review,
    ReviewStore,
)
from analyst_app.theme import SERIES, STATUS_COLOURS

NOW = datetime(2023, 6, 1, 12, 0, tzinfo=UTC)


# --- Agreement --------------------------------------------------------------


def test_an_analyst_confirming_a_block_agrees_with_the_platform() -> None:
    review = Review(1, VERDICT_FRAUD, "ana", platform_decision="block")

    assert review.is_fraud
    assert review.agreed_with_platform


def test_an_analyst_clearing_a_block_disagrees() -> None:
    """The case worth counting: the platform stopped a real customer."""
    review = Review(1, VERDICT_LEGITIMATE, "ana", platform_decision="block")

    assert review.agreed_with_platform is False


def test_a_review_that_turns_out_to_be_fraud_disagrees_with_the_queue() -> None:
    """Sent for review rather than blocked, and it was fraud after all."""
    review = Review(1, VERDICT_FRAUD, "ana", platform_decision="review")

    assert review.agreed_with_platform is False


# --- Colour choices ---------------------------------------------------------


def test_decision_colours_are_semantic_and_distinct() -> None:
    assert set(STATUS_COLOURS) == {"allow", "review", "block"}
    assert len(set(STATUS_COLOURS.values())) == 3


def test_series_colours_are_a_fixed_order_never_cycled() -> None:
    """A filter that drops a series must not repaint the survivors."""
    assert SERIES == ["#2a78d6", "#eb6834", "#1baf7a"]


def test_charts_never_get_a_second_y_axis() -> None:
    """Two measures of different scale get two charts, not two scales."""
    import plotly.graph_objects as go

    from analyst_app.theme import style

    figure = style(go.Figure(), y_title="payments")

    assert "yaxis2" not in figure.layout
    assert figure.layout.yaxis.title.text == "payments"


def test_an_untitled_chart_has_no_title_object() -> None:
    """Passing title=None leaves Plotly rendering the string "undefined"."""
    import plotly.graph_objects as go

    from analyst_app.theme import style

    figure = style(go.Figure())

    assert figure.layout.title.text is None


# --- The feedback loop ------------------------------------------------------


class _FakeCaseStore:
    def __init__(self) -> None:
        self.added = []

    def add_cases(self, cases):
        self.added.extend(cases)
        return len(cases)


@pytest.fixture
def review_store():
    import os

    import pytest as _pytest

    try:
        from genai.case_store import connect
    except ImportError:  # pragma: no cover
        _pytest.skip("psycopg not installed")

    connection = connect(
        host=os.environ.get("PGVECTOR_HOST", "localhost"),
        port=int(os.environ.get("PGVECTOR_PORT", "5432")),
        database=os.environ.get("PGVECTOR_DB", "fraud_cases"),
        user=os.environ.get("PGVECTOR_USER", "fraud"),
        password=os.environ.get("PGVECTOR_PASSWORD", "fraud_local_dev_only"),
    )
    with connection.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS analyst_reviews_test")
    connection.commit()

    case_store = _FakeCaseStore()
    store = ReviewStore(connection, case_store, table="analyst_reviews_test")
    store.create_schema()
    yield store, case_store

    with connection.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS analyst_reviews_test")
    connection.commit()
    connection.close()


@pytest.mark.needs_pgvector
def test_a_review_is_recorded_and_becomes_a_confirmed_case(review_store) -> None:
    """The feedback loop in one test.

    An analyst's confirmation arrives in minutes; a chargeback takes weeks.
    Both are real outcomes, so both belong in the case store - marked with how
    they were resolved so nobody mistakes a judgement for a settled dispute.
    """
    store, case_store = review_store

    store.record(
        Review(5001, VERDICT_FRAUD, "ana", platform_decision="review", note="card testing"),
        case_facts={
            "features": {"amount": 420.0, "card_is_new": 1.0},
            "card_token": "b" * 32,
            "decided_at": NOW,
        },
    )

    assert store.reviewed_ids() == {5001}
    assert len(case_store.added) == 1
    added = case_store.added[0]
    assert added.is_fraud is True
    assert added.resolution == "analyst_review"
    assert "first payment seen on this card" in added.description


@pytest.mark.needs_pgvector
def test_reviewed_cases_leave_the_queue(review_store) -> None:
    store, _ = review_store

    store.record(Review(5002, VERDICT_LEGITIMATE, "ana", platform_decision="block"), {})

    assert 5002 in store.reviewed_ids()


@pytest.mark.needs_pgvector
def test_reviewing_the_same_case_twice_updates_rather_than_duplicates(review_store) -> None:
    store, _ = review_store

    store.record(Review(5003, VERDICT_FRAUD, "ana", platform_decision="review"), {})
    store.record(Review(5003, VERDICT_LEGITIMATE, "bob", platform_decision="review"), {})

    recent = store.recent()
    assert len([r for r in recent if r.transaction_id == 5003]) == 1
    assert recent[0].verdict == VERDICT_LEGITIMATE


@pytest.mark.needs_pgvector
def test_agreement_rate_is_reported(review_store) -> None:
    """Near 100% may mean the queue is too easy to be worth a human's time."""
    store, _ = review_store

    store.record(Review(5004, VERDICT_FRAUD, "ana", platform_decision="block"), {})
    store.record(Review(5005, VERDICT_LEGITIMATE, "ana", platform_decision="block"), {})

    assert store.agreement_rate() == pytest.approx(0.5)


@pytest.mark.needs_pgvector
def test_a_failing_case_store_does_not_lose_the_review(review_store) -> None:
    """The analyst's decision is the thing that must never be dropped."""
    store, _ = review_store

    class BrokenCaseStore:
        def add_cases(self, cases):
            raise ConnectionError("pgvector is down")

    store._case_store = BrokenCaseStore()
    store.record(
        Review(5006, VERDICT_FRAUD, "ana", platform_decision="review"),
        case_facts={"features": {"amount": 10.0}},
    )

    assert 5006 in store.reviewed_ids()
