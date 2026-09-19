"""Tests for the shared feature definitions.

These pin down the semantics that both the online and the offline path have to
honour. If one of these assertions changes, the model has to be retrained.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from features.definitions import (
    FEATURE_NAMES,
    MAX_TRACKED_DEVICES,
    NO_HISTORY,
    CardState,
    compute_and_update,
    compute_features,
    update_state,
)

NOON = datetime(2023, 6, 1, 12, 0, 0, tzinfo=UTC)


def _history(make_event, offsets_and_amounts, **kwargs) -> CardState:
    """Fold a list of (minutes_before_noon, amount) payments into a state."""
    state = CardState()
    for minutes_before, amount in offsets_and_amounts:
        event = make_event(at=NOON - timedelta(minutes=minutes_before), amount=amount, **kwargs)
        state = update_state(state, event)
    return state


def test_feature_vector_matches_the_declared_names(make_event) -> None:
    """The model is trained on FEATURE_NAMES in order; nothing may be missing."""
    features = compute_features(CardState(), make_event())

    assert list(features) == FEATURE_NAMES
    assert all(isinstance(value, float) for value in features.values())


def test_a_brand_new_card_has_no_history(make_event) -> None:
    features = compute_features(CardState(), make_event(amount=50.0))

    assert features["card_is_new"] == 1.0
    assert features["card_txn_count_lifetime"] == 0.0
    assert features["seconds_since_card_last_txn"] == NO_HISTORY
    # No baseline to compare against, so the ratio is neutral rather than huge.
    assert features["amount_to_card_avg_ratio"] == 1.0


def test_the_current_payment_counts_towards_its_own_windows(make_event) -> None:
    """A card's first payment is one payment in the last ten minutes, not zero.

    This is also why the scoring API cannot just read a cached count: the
    number it needs does not exist until the payment arrives.
    """
    features = compute_features(CardState(), make_event())

    assert features["card_txn_count_10m"] == 1.0
    assert features["card_txn_count_1h"] == 1.0
    assert features["card_txn_count_24h"] == 1.0


def test_windows_only_count_events_inside_them(make_event) -> None:
    state = _history(
        make_event,
        [
            (2, 10.0),  # 2 minutes ago  -> in 10m, 1h and 24h
            (30, 20.0),  # 30 minutes ago -> in 1h and 24h
            (300, 30.0),  # 5 hours ago    -> in 24h only
            (60 * 30, 40.0),  # 30 hours ago   -> aged out entirely
        ],
    )

    features = compute_features(state, make_event(amount=5.0))

    # +1 in each case for the payment being scored.
    assert features["card_txn_count_10m"] == 2.0
    assert features["card_txn_count_1h"] == 3.0
    assert features["card_txn_count_24h"] == 4.0
    assert features["card_amount_sum_24h"] == pytest.approx(10.0 + 20.0 + 30.0 + 5.0)


def test_an_event_exactly_on_the_boundary_is_inside_the_window(make_event) -> None:
    """The window is half-open: (now - window, now].

    An arbitrary choice, but it has to be the same choice in both paths, so it
    is written down and tested rather than left to whichever implementation
    happens to be running.
    """
    state = _history(make_event, [(10, 10.0)])

    features = compute_features(state, make_event())

    assert features["card_txn_count_10m"] == 2.0


def test_velocity_burst_is_visible(make_event) -> None:
    """The signal this whole pipeline exists to catch."""
    state = _history(make_event, [(m, 200.0) for m in (1, 2, 3, 4, 5)])

    features = compute_features(state, make_event(amount=200.0))

    assert features["card_txn_count_10m"] == 6.0
    assert features["seconds_since_card_last_txn"] == 60.0


def test_amount_is_compared_to_the_cards_own_baseline(make_event) -> None:
    state = _history(make_event, [(120, 100.0), (240, 100.0)])

    features = compute_features(state, make_event(amount=500.0))

    assert features["card_amount_avg_lifetime"] == pytest.approx(100.0)
    assert features["amount_to_card_avg_ratio"] == pytest.approx(5.0)


def test_a_new_device_is_only_new_once_the_card_has_history(make_event) -> None:
    """On a card's first payment every device is unfamiliar, which is no signal."""
    first = make_event(device_info="iOS Device")

    assert compute_features(CardState(), first)["card_new_device"] == 0.0

    state = update_state(CardState(), first)
    same_device = compute_features(state, make_event(device_info="iOS Device"))
    other_device = compute_features(state, make_event(device_info="Windows"))

    assert same_device["card_new_device"] == 0.0
    assert other_device["card_new_device"] == 1.0


def test_a_new_recipient_email_domain_is_flagged(make_event) -> None:
    state = update_state(CardState(), make_event(r_emaildomain="gmail.com"))

    known = compute_features(state, make_event(r_emaildomain="gmail.com"))
    unknown = compute_features(state, make_event(r_emaildomain="anonymous.com"))

    assert known["card_new_email_domain"] == 0.0
    assert unknown["card_new_email_domain"] == 1.0


def test_missing_device_is_not_treated_as_a_new_device(make_event) -> None:
    """Three quarters of transactions have no identity record at all."""
    state = _history(make_event, [(10, 10.0)], device_info="Windows")

    features = compute_features(state, make_event(device_info=None))

    assert features["card_new_device"] == 0.0
    assert features["has_identity"] == 0.0


def test_stateless_features_describe_the_payment(make_event) -> None:
    night = make_event(at=datetime(2023, 6, 4, 3, 30, tzinfo=UTC), amount=99.0)

    features = compute_features(CardState(), night)

    assert features["hour_of_day"] == 3.0
    assert features["is_night"] == 1.0
    assert features["day_of_week"] == 6.0  # a Sunday
    assert features["amount"] == 99.0
    assert features["amount_log"] == pytest.approx(math.log1p(99.0))


def test_update_state_does_not_mutate_its_input(make_event) -> None:
    """Callers compute features from the old state, then update; order must not matter."""
    state = CardState()

    updated = update_state(state, make_event(amount=10.0))

    assert state.txn_count_lifetime == 0
    assert updated.txn_count_lifetime == 1


def test_state_forgets_events_older_than_the_longest_window(make_event) -> None:
    """Otherwise a busy card's Redis entry grows forever."""
    state = _history(make_event, [(60 * 48, 10.0), (60 * 30, 10.0), (5, 10.0)])

    # Both older events have aged out; only the most recent one is still held.
    assert len(state.recent_events) == 1
    # Lifetime totals still remember everything.
    assert state.txn_count_lifetime == 3


def test_tracked_devices_are_bounded(make_event) -> None:
    state = CardState()
    for i in range(MAX_TRACKED_DEVICES + 10):
        state = update_state(state, make_event(device_info=f"device-{i}"))

    assert len(state.known_devices) == MAX_TRACKED_DEVICES


def test_state_survives_a_redis_round_trip(make_event) -> None:
    state = _history(make_event, [(5, 10.0), (60, 20.0)], device_info="Windows")

    restored = CardState.from_json(state.to_json())

    assert restored == state


def test_corrupt_state_is_treated_as_a_new_card() -> None:
    """The safe failure mode: an unfamiliar card biases towards review."""
    assert CardState.from_json("not json").is_empty
    assert CardState.from_json(None).is_empty
    assert CardState.from_json("").is_empty


def test_compute_and_update_returns_both(make_event) -> None:
    features, state = compute_and_update(CardState(), make_event(amount=25.0))

    assert features["card_is_new"] == 1.0
    assert state.txn_count_lifetime == 1
    assert state.amount_sum_lifetime == 25.0


def test_out_of_order_events_do_not_move_the_clock_backwards(make_event) -> None:
    """Kafka keeps a card's payments ordered, but a replay can still surprise us."""
    state = update_state(CardState(), make_event(at=NOON))

    state = update_state(state, make_event(at=NOON - timedelta(hours=1)))

    assert state.last_event_epoch == NOON.timestamp()


def test_a_window_never_counts_events_from_the_future(make_event) -> None:
    """Point-in-time correctness, defended in the function itself.

    Payments normally reach the state object in order, so this cannot happen -
    but "cannot happen" is how leaks get in. A payment scored late must see
    only what preceded it.
    """
    state = _history(make_event, [(-5, 500.0), (5, 10.0)])  # one 5 minutes later

    features = compute_features(state, make_event())

    # Only the earlier payment and the one being scored.
    assert features["card_txn_count_10m"] == 2.0
    assert features["card_amount_sum_24h"] == pytest.approx(10.0 + 100.0)
