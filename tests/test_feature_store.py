"""Tests for the Redis online feature store.

Every test runs against fakeredis: no server, no Docker, no flakiness.
"""

from __future__ import annotations

import pytest

from common.pii import card_token
from features.definitions import CardState, update_state
from features.store import FeatureStore, state_key

SALT = "c" * 64
CARD_KEY = "13926|315|gmail.com"


@pytest.fixture
def store():
    import fakeredis

    return FeatureStore(fakeredis.FakeStrictRedis(decode_responses=True), salt=SALT, ttl_seconds=60)


def test_unknown_card_returns_empty_state(store) -> None:
    """A card the platform has never seen still has to be scoreable."""
    assert store.get_state("never|seen|before").is_empty


def test_state_round_trips(store, make_event) -> None:
    state = update_state(CardState(), make_event(amount=42.0, device_info="Windows"))

    store.set_state(CARD_KEY, state)

    restored = store.get_state(CARD_KEY)
    assert restored.txn_count_lifetime == 1
    assert restored.amount_sum_lifetime == 42.0
    assert restored.known_devices == ["Windows"]


def test_redis_keys_contain_the_token_not_the_card(store) -> None:
    """The point of the tokenisation: a Redis dump has no card identities in it."""
    store.set_state(CARD_KEY, CardState(txn_count_lifetime=1))

    keys = list(store._client.scan_iter(match="fp:card:*"))
    assert keys == [state_key(card_token(CARD_KEY, SALT))]
    assert not any("gmail.com" in key for key in keys)


def test_state_expires(store) -> None:
    """Stale history must not be scored against; TTL is the safety net."""
    store.set_state(CARD_KEY, CardState(txn_count_lifetime=1))

    ttl = store._client.ttl(state_key(store.token_for(CARD_KEY)))
    assert 0 < ttl <= 60


def test_set_many_writes_every_card_in_one_pipeline(store) -> None:
    states = [(f"{i}|1|gmail.com", CardState(txn_count_lifetime=i + 1)) for i in range(50)]

    written = store.set_many(states)

    assert written == 50
    assert store.card_count() == 50
    assert store.get_state("7|1|gmail.com").txn_count_lifetime == 8


def test_set_many_with_nothing_to_write_is_a_no_op(store) -> None:
    assert store.set_many([]) == 0


def test_corrupt_entry_does_not_break_scoring(store) -> None:
    store._client.set(state_key(store.token_for(CARD_KEY)), "{not json")

    assert store.get_state(CARD_KEY).is_empty
