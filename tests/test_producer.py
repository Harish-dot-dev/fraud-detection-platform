"""Tests for the Kafka replay producer.

None of these need a broker: ``replay()`` takes an injected sink and an
injected clock, so ordering, keying and pacing can all be checked in
milliseconds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd
import pytest

from common.timeline import DEFAULT_EPOCH
from producer.replay import (
    FIXTURE_TRANSACTIONS,
    ListSink,
    iter_payment_events,
    load_source,
    replay,
    resolve_source_paths,
)


@dataclass
class FakeClock:
    """A clock that only moves when the code under test sleeps."""

    t: float = 0.0
    sleeps: list[float] = field(default_factory=list)

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds

    def advance(self, seconds: float) -> None:
        """Simulate time passing for a reason other than sleeping."""
        self.t += seconds


def _events(dts: list[int], card_keys: list[str] | None = None):
    """Build minimal payment events at the given TransactionDT offsets."""
    card_keys = card_keys or ["1|2|gmail.com"] * len(dts)
    frame = pd.DataFrame(
        {
            "TransactionID": range(1, len(dts) + 1),
            "TransactionDT": dts,
            "TransactionAmt": [10.0] * len(dts),
            "card1": [int(k.split("|")[0]) for k in card_keys],
            "addr1": [float(k.split("|")[1]) for k in card_keys],
            "P_emaildomain": [k.split("|")[2] for k in card_keys],
        }
    )
    return list(iter_payment_events(frame, epoch=DEFAULT_EPOCH))


def test_events_are_emitted_in_transaction_order() -> None:
    sink = ListSink()

    replay(iter(_events([300, 100, 200])), sink, speedup=0)

    sent = [json.loads(value)["transaction_dt"] for _, value in sink.messages]
    # iter_payment_events preserves the frame order; load_source is what sorts,
    # so this documents that replay itself does not reorder.
    assert sent == [300, 100, 200]


def test_load_source_sorts_by_transaction_dt() -> None:
    transactions, identity_index = load_source(FIXTURE_TRANSACTIONS, limit=50)

    assert transactions["TransactionDT"].is_monotonic_increasing
    assert len(transactions) == 50
    assert identity_index == {}


def test_kafka_key_is_the_card_identity_proxy() -> None:
    sink = ListSink()
    keys = ["111|222|gmail.com", "333|444|yahoo.com"]

    replay(iter(_events([100, 200], card_keys=keys)), sink, speedup=0)

    assert [key for key, _ in sink.messages] == keys


def test_max_events_stops_the_replay() -> None:
    sink = ListSink()

    stats = replay(iter(_events([100, 200, 300, 400])), sink, speedup=0, max_events=2)

    assert stats.events_sent == 2
    assert len(sink.messages) == 2


def test_speedup_zero_never_sleeps() -> None:
    clock = FakeClock()
    sink = ListSink()

    replay(iter(_events([0, 3600, 7200])), sink, speedup=0, sleep=clock.sleep, now=clock.now)

    assert clock.sleeps == []


def test_gaps_between_events_are_preserved_and_divided_by_the_speedup() -> None:
    clock = FakeClock()
    sink = ListSink()

    # Events 60 dataset-seconds apart, replayed at 60x -> one second each.
    replay(iter(_events([0, 60, 120])), sink, speedup=60, sleep=clock.sleep, now=clock.now)

    assert clock.sleeps == pytest.approx([1.0, 1.0])


class _SlowSink(ListSink):
    """A sink that takes time to send, to test schedule drift."""

    def __init__(self, clock: FakeClock, cost: float) -> None:
        super().__init__()
        self._clock = clock
        self._cost = cost

    def send(self, key: str, value: str) -> None:
        super().send(key, value)
        self._clock.advance(self._cost)


def test_pacing_does_not_drift_when_sending_is_slow() -> None:
    """Sleeps are computed against a fixed schedule, not chained together.

    If the replay slept for the full gap after every send, a slow broker would
    stretch the whole run. Scheduling from the start time means a slow send
    eats into the next sleep instead.
    """
    clock = FakeClock()
    sink = _SlowSink(clock, cost=0.4)

    replay(iter(_events([0, 60, 120])), sink, speedup=60, sleep=clock.sleep, now=clock.now)

    # Each send costs 0.4s of the 1.0s budget, so each sleep is 0.6s...
    assert clock.sleeps == pytest.approx([0.6, 0.6])
    # ...and the third event still goes out on schedule at t=2.0 (the extra
    # 0.4 is that last send itself). Chaining sleeps instead would have put it
    # at t=2.8 and the gap would grow with every event.
    assert clock.now() == pytest.approx(2.4)


def test_stats_describe_the_run() -> None:
    sink = ListSink()

    # Two days apart in dataset time.
    stats = replay(iter(_events([0, 172_800])), sink, speedup=0)

    assert stats.events_sent == 2
    assert stats.dataset_seconds == 172_800
    assert "2.00 days" in stats.describe()


def test_source_falls_back_to_the_fixture_when_the_dataset_is_absent() -> None:
    """A fresh clone with no Kaggle data must still be able to produce."""
    transactions_path, identity_path, is_real = resolve_source_paths(prefer_raw=False)

    assert transactions_path == FIXTURE_TRANSACTIONS
    assert identity_path.exists()
    assert is_real is False


def test_identity_is_joined_onto_matching_transactions() -> None:
    transactions, identity_index = load_source(
        FIXTURE_TRANSACTIONS, FIXTURE_TRANSACTIONS.parent / "train_identity_sample.csv"
    )
    events = list(iter_payment_events(transactions, identity_index, epoch=DEFAULT_EPOCH))

    with_device = [e for e in events if e.device_type is not None]
    assert 0 < len(with_device) < len(events)
    assert len(with_device) == len(identity_index)


# --- which Kafka listener the producer talks to -----------------------------


class _Marker:
    """Stands in for /.dockerenv."""

    def __init__(self, present: bool) -> None:
        self._present = present

    def exists(self) -> bool:
        return self._present


def test_bootstrap_servers_default_differs_inside_a_container(monkeypatch) -> None:
    """The broker publishes two listeners and only one works from each side.

    `make produce` used to run on the host against the published port, which
    is the listener whose reachability depends on the host's Docker
    networking rather than on the broker being up. It now runs in the
    container like every other pipeline step, so the default has to follow.
    """
    from producer import replay

    monkeypatch.setattr(replay, "DOCKER_MARKER", _Marker(False))
    assert replay.default_bootstrap_servers() == "localhost:29092"

    monkeypatch.setattr(replay, "DOCKER_MARKER", _Marker(True))
    assert replay.default_bootstrap_servers() == "kafka:9092"


def test_limit_bounds_the_read_not_just_the_emit(tmp_path) -> None:
    """--limit used to be applied only to the emit loop.

    The whole 652 MB transaction file and all 144,233 identity rows were read
    first, so a five-thousand-event smoke test cost the same memory as a full
    run. Invisible on the 1000-row fixture, fatal on the real dataset.
    """
    import pandas as pd

    from producer.replay import load_source

    transactions = pd.DataFrame(
        {
            "TransactionID": range(1000, 1100),
            "TransactionDT": range(100),
            "TransactionAmt": [10.0] * 100,
        }
    )
    identity = pd.DataFrame({"TransactionID": range(1000, 1100), "DeviceType": ["desktop"] * 100})
    transactions.to_csv(tmp_path / "t.csv", index=False)
    identity.to_csv(tmp_path / "i.csv", index=False)

    loaded, index = load_source(tmp_path / "t.csv", tmp_path / "i.csv", limit=10)

    assert len(loaded) == 10, "the transaction read must respect the limit"
    assert len(index) == 10, "only the identities of the loaded rows should be indexed"
