"""Tests for the online (streaming) feature update.

The Spark plumbing is thin; what matters is the fold that turns a batch of
payments into card state, and that runs perfectly well without Spark.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from features.definitions import CardState, compute_features, update_state
from features.store import FeatureStore
from streaming.features import RedisConfig, row_to_event, update_states

SALT = "d" * 64
START = datetime(2023, 6, 1, 9, 0, 0, tzinfo=UTC)


@pytest.fixture
def store():
    import fakeredis

    return FeatureStore(fakeredis.FakeStrictRedis(decode_responses=True), salt=SALT)


def test_a_batch_is_folded_into_card_state(store, make_event) -> None:
    events = [make_event(at=START + timedelta(minutes=i), amount=10.0 * (i + 1)) for i in range(3)]

    written = update_states(events, store)

    state = store.get_state(events[0].card_key)
    assert written == 1  # one card
    assert state.txn_count_lifetime == 3
    assert state.amount_sum_lifetime == pytest.approx(60.0)


def test_each_card_gets_its_own_state(store, make_event) -> None:
    events = [
        make_event(card_key="111|1|gmail.com", amount=10.0),
        make_event(card_key="222|2|yahoo.com", amount=20.0),
        make_event(card_key="111|1|gmail.com", amount=30.0),
    ]

    written = update_states(events, store)

    assert written == 2
    assert store.get_state("111|1|gmail.com").txn_count_lifetime == 2
    assert store.get_state("222|2|yahoo.com").txn_count_lifetime == 1


def test_state_carries_over_between_batches(store, make_event) -> None:
    """A micro-batch is not a session; yesterday's history still counts."""
    update_states([make_event(at=START, amount=10.0)], store)

    update_states([make_event(at=START + timedelta(minutes=5), amount=20.0)], store)

    state = store.get_state("13926|315|gmail.com")
    assert state.txn_count_lifetime == 2
    assert len(state.recent_events) == 2


def test_a_card_is_read_once_per_batch_however_many_payments_it_has(store, make_event) -> None:
    """Twenty payments on one card cost one Redis read, not twenty."""
    reads = {"n": 0}
    original = store.get_state

    def counting_get_state(card_key: str):
        reads["n"] += 1
        return original(card_key)

    store.get_state = counting_get_state  # type: ignore[method-assign]
    events = [make_event(at=START + timedelta(seconds=i)) for i in range(20)]

    update_states(events, store)

    assert reads["n"] == 1


def test_the_result_matches_folding_by_hand(store, make_event) -> None:
    """The streaming path must agree with the shared definitions exactly."""
    events = [
        make_event(at=START + timedelta(minutes=i), amount=float(i + 1), device_info="Windows")
        for i in range(5)
    ]

    update_states(events, store)

    expected = CardState()
    for event in events:
        expected = update_state(expected, event)
    assert store.get_state(events[0].card_key) == expected


def test_features_computed_after_a_batch_see_that_batch(store, make_event) -> None:
    """The point of the whole exercise: the next payment sees the last one."""
    burst = [make_event(at=START + timedelta(seconds=30 * i), amount=100.0) for i in range(4)]
    update_states(burst, store)

    next_payment = make_event(at=START + timedelta(minutes=3), amount=100.0)
    features = compute_features(store.get_state(next_payment.card_key), next_payment)

    assert features["card_txn_count_10m"] == 5.0
    assert features["card_is_new"] == 0.0


def test_row_to_event_reads_the_fields_the_fold_needs() -> None:
    """Spark Rows are dict-like; this is the boundary between the two worlds."""
    row = {
        "transaction_id": 42,
        "event_time": START,
        "transaction_dt": 1000,
        "card_key": "1|2|gmail.com",
        "amount": 12.5,
        "device_info": "iOS Device",
        "r_emaildomain": "gmail.com",
    }

    event = row_to_event(row)

    assert event.transaction_id == 42
    assert event.amount == 12.5
    assert event.device_info == "iOS Device"


def test_redis_config_is_picklable() -> None:
    """It is shipped to every Spark executor, so it must not hold a client."""
    import pickle

    config = RedisConfig(host="redis", port=6379, salt=SALT, ttl_seconds=60)

    assert pickle.loads(pickle.dumps(config)) == config
