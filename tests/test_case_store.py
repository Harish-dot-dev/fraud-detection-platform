"""Tests for the pgvector case store.

These need a real Postgres with the vector extension - the SQL is the thing
being tested, and a fake would test nothing. Marked `needs_pgvector`; the
Compose `ai` profile provides one.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest

from genai.case_store import CaseStore, PastCase, connect, describe_case
from genai.embeddings import HashingEmbedder

pytestmark = pytest.mark.needs_pgvector

NOW = datetime(2023, 6, 1, 12, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def store():
    connection = connect(
        host=os.environ.get("PGVECTOR_HOST", "localhost"),
        port=int(os.environ.get("PGVECTOR_PORT", "5432")),
        database=os.environ.get("PGVECTOR_DB", "fraud_cases"),
        user=os.environ.get("PGVECTOR_USER", "fraud"),
        password=os.environ.get("PGVECTOR_PASSWORD", "fraud_local_dev_only"),
    )
    # A table of its own, so the tests never touch a loaded case store.
    with connection.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS fraud_cases_test")
    connection.commit()

    import genai.case_store as module

    original = module.TABLE
    module.TABLE = "fraud_cases_test"
    module.SCHEMA = module.SCHEMA.replace(original, "fraud_cases_test")

    case_store = CaseStore(connection, HashingEmbedder())
    case_store.create_schema()
    yield case_store

    with connection.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS fraud_cases_test")
    connection.commit()
    connection.close()
    module.TABLE = original


def _case(transaction_id: int, amount: float, is_fraud: bool, **features) -> PastCase:
    return PastCase(
        transaction_id=transaction_id,
        card_token=f"{transaction_id:032d}",
        occurred_at=NOW + timedelta(minutes=transaction_id),
        amount=amount,
        decision="block" if is_fraud else "allow",
        is_fraud=is_fraud,
        description=describe_case(amount, "block" if is_fraud else "allow", features),
    )


@pytest.fixture(scope="module", autouse=True)
def seeded(store):
    store.add_cases(
        [
            _case(1, 2400.0, True, card_is_new=1.0, card_new_device=1.0),
            _case(2, 2600.0, True, card_is_new=1.0, card_new_device=1.0),
            _case(3, 25.0, False, card_txn_count_24h=3.0),
            _case(4, 30.0, False, card_txn_count_24h=2.0),
        ]
    )
    return store


def test_cases_are_stored(store) -> None:
    assert store.count() == 4


def test_similar_cases_come_back_in_order(store) -> None:
    """The panel an analyst reads: payments that looked like this one."""
    query = describe_case(2500.0, "block", {"card_is_new": 1.0, "card_new_device": 1.0})

    results = store.find_similar(query, top_k=2)

    assert [case.transaction_id for case in results] == [1, 2]
    assert all(case.is_fraud for case in results)
    assert results[0].similarity >= results[1].similarity


def test_a_small_table_returns_everything_asked_for(store) -> None:
    """The regression test for an approximate-index trap.

    IVFFlat searches one list by default. Created over a handful of rows it
    silently returns fewer neighbours than requested - a three-row table once
    returned two results for a request for three. The index is now only built
    once there is enough data to need it.
    """
    results = store.find_similar(describe_case(100.0, "allow", {}), top_k=4)

    assert len(results) == 4


def test_a_payment_never_retrieves_itself(store) -> None:
    """An identical match is a perfect, useless neighbour - and during
    evaluation it would quietly turn retrieval quality into 100%."""
    query = describe_case(2400.0, "block", {"card_is_new": 1.0, "card_new_device": 1.0})

    results = store.find_similar(query, top_k=3, exclude_transaction_id=1)

    assert 1 not in [case.transaction_id for case in results]


def test_reloading_a_case_updates_it_rather_than_duplicating(store) -> None:
    updated = _case(1, 2400.0, True, card_is_new=1.0)
    updated.resolution = "analyst_review"

    store.add_cases([updated])

    assert store.count() == 4
    results = store.find_similar(updated.description, top_k=1)
    assert results[0].resolution == "analyst_review"


def test_the_index_waits_until_there_is_enough_data(store) -> None:
    assert store.ensure_index() is False
    assert store.ensure_index(min_rows=2) is True


def test_a_retrieved_case_reads_as_one_line(store) -> None:
    case = store.find_similar(describe_case(2500.0, "block", {"card_is_new": 1.0}), top_k=1)[0]

    line = case.summary_line()
    assert "confirmed fraud" in line
    assert "2023-06-01" in line
