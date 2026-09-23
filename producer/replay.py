"""Replay historical payments onto Kafka in their original time order.

The IEEE-CIS data is a static CSV, but the platform is meant to be a streaming
system, so the producer turns the file back into a stream: rows are emitted in
``TransactionDT`` order, with the real gaps between them preserved and divided
by a speed-up factor. At the default ``REPLAY_SPEEDUP=3600`` one hour of
history takes one second, so the ~six month dataset replays in about an hour.

    make produce                      # fixture or data/raw, whichever exists
    make produce ARGS="--limit 500 --dry-run"

Design notes
------------
* **Ordering.** Rows are sorted by ``TransactionDT`` and keyed by the card
  identity proxy, so Kafka's per-partition ordering keeps a single card's
  payments in sequence. Velocity features depend on that.
* **Pacing.** Sleep is computed against a fixed schedule rather than by adding
  up per-event sleeps, so the replay does not drift when a send is slow.
* **The sink is injected.** ``replay()`` writes to anything with a ``send``
  method, which is what lets the tests verify ordering, keys and pacing
  without a Kafka broker anywhere in sight.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from common.config import REPO_ROOT, get_settings
from common.events import PaymentEvent, payment_event_from_row
from common.timeline import parse_epoch

logger = logging.getLogger("producer")

RAW_TRANSACTIONS = REPO_ROOT / "data" / "raw" / "train_transaction.csv"
RAW_IDENTITY = REPO_ROOT / "data" / "raw" / "train_identity.csv"
FIXTURE_TRANSACTIONS = REPO_ROOT / "tests" / "fixtures" / "train_transaction_sample.csv"
FIXTURE_IDENTITY = REPO_ROOT / "tests" / "fixtures" / "train_identity_sample.csv"

# Sleeping for less than this is not worth the syscall; the schedule absorbs it.
MIN_SLEEP_SECONDS = 0.001

# How many identity rows to hold in memory at once while indexing. The real
# file is 144,233 rows wide by 41 columns; reading it whole and converting it
# to Python dicts costs more memory than everything else the producer does put
# together. See load_source.
IDENTITY_CHUNK_ROWS = 50_000


class EventSink(Protocol):
    """Anywhere a payment event can be written."""

    def send(self, key: str, value: str) -> None: ...

    def flush(self) -> None: ...


@dataclass
class ListSink:
    """Collects events in memory. Used by the tests and by ``--dry-run``."""

    messages: list[tuple[str, str]] = field(default_factory=list)

    def send(self, key: str, value: str) -> None:
        self.messages.append((key, value))

    def flush(self) -> None:
        return None


class KafkaSink:
    """Publishes to a Kafka topic via confluent-kafka."""

    def __init__(self, bootstrap_servers: str, topic: str) -> None:
        # Imported here so the module stays importable (and testable) on a
        # machine with no Kafka client installed.
        from confluent_kafka import Producer

        self.topic = topic
        self.delivery_failures = 0
        self._producer = Producer(
            {
                "bootstrap.servers": bootstrap_servers,
                "linger.ms": 5,
                "compression.type": "lz4",
                # Ordering per partition must survive a retry.
                "enable.idempotence": True,
                "acks": "all",
            }
        )

    def _on_delivery(self, err: Any, _msg: Any) -> None:
        if err is not None:
            self.delivery_failures += 1
            logger.error("delivery failed: %s", err)

    def send(self, key: str, value: str) -> None:
        self._producer.produce(
            self.topic, key=key.encode(), value=value.encode(), on_delivery=self._on_delivery
        )
        # Serve delivery callbacks without blocking.
        self._producer.poll(0)

    def flush(self) -> None:
        self._producer.flush(30)


@dataclass
class ReplayStats:
    """What a replay actually did - printed at the end of a run."""

    events_sent: int = 0
    first_event_time: datetime | None = None
    last_event_time: datetime | None = None
    wall_seconds: float = 0.0

    @property
    def dataset_seconds(self) -> float:
        if self.first_event_time is None or self.last_event_time is None:
            return 0.0
        return (self.last_event_time - self.first_event_time).total_seconds()

    @property
    def events_per_second(self) -> float:
        return self.events_sent / self.wall_seconds if self.wall_seconds > 0 else 0.0

    def describe(self) -> str:
        return (
            f"sent {self.events_sent} events "
            f"covering {self.dataset_seconds / 86_400:.2f} days of history "
            f"in {self.wall_seconds:.1f}s ({self.events_per_second:.0f} events/s)"
        )


def resolve_source_paths(prefer_raw: bool = True) -> tuple[Path, Path, bool]:
    """Pick the real dataset if it is present, otherwise the synthetic fixture.

    Returns ``(transactions_path, identity_path, is_real_data)``.
    """
    if prefer_raw and RAW_TRANSACTIONS.exists():
        return RAW_TRANSACTIONS, RAW_IDENTITY, True
    return FIXTURE_TRANSACTIONS, FIXTURE_IDENTITY, False


def load_source(
    transactions_path: Path, identity_path: Path | None = None, limit: int | None = None
) -> tuple[pd.DataFrame, dict[int, dict[str, Any]]]:
    """Read the transaction file and index the identity file by TransactionID.

    The identity file is indexed into a dict rather than merged: a join would
    widen every row by 40 mostly-empty columns, and the producer only needs a
    lookup.
    """
    transactions = pd.read_csv(transactions_path, nrows=limit)
    transactions = transactions.sort_values("TransactionDT", kind="stable")

    identity_index: dict[int, dict[str, Any]] = {}
    if identity_path is not None and identity_path.exists():
        # Only the identities belonging to the transactions actually loaded,
        # read in chunks so the file never lands in memory whole.
        #
        # This used to read the entire identity file and call
        # to_dict("records") on it, which builds one Python dict of ~41 keys
        # per row - 144,233 of them on the real dataset, about 1.5-2 GB at
        # peak, and it happened whatever --limit was set to. On the 1000-row
        # test fixture that is invisible; on the real file it gets the process
        # killed by the OOM killer before a single event is produced.
        wanted = set(transactions["TransactionID"].astype("int64"))
        for chunk in pd.read_csv(identity_path, chunksize=IDENTITY_CHUNK_ROWS):
            matching = chunk[chunk["TransactionID"].isin(wanted)]
            identity_index.update(
                {int(row["TransactionID"]): row for row in matching.to_dict(orient="records")}
            )
    return transactions, identity_index


def iter_payment_events(
    transactions: pd.DataFrame,
    identity_index: dict[int, dict[str, Any]] | None = None,
    epoch: datetime | None = None,
) -> Iterator[PaymentEvent]:
    """Yield payment events in ``TransactionDT`` order."""
    identity_index = identity_index or {}
    epoch = epoch or parse_epoch(get_settings().synthetic_epoch)

    for row in transactions.to_dict(orient="records"):
        transaction_id = int(row["TransactionID"])
        yield payment_event_from_row(row, identity_index.get(transaction_id), epoch=epoch)


def replay(
    events: Iterator[PaymentEvent],
    sink: EventSink,
    speedup: float = 3600.0,
    max_events: int = 0,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
    progress_every: int = 1000,
) -> ReplayStats:
    """Send events to ``sink``, preserving the gaps between them.

    Args:
        events: payment events, already in time order.
        sink: anything with ``send(key, value)`` and ``flush()``.
        speedup: dataset seconds per wall-clock second. ``0`` means as fast as
            possible, which is what the tests and load generation use.
        max_events: stop after this many events (``0`` = no limit).
        sleep / now: injected for testing; a test must never actually wait.
        progress_every: log a progress line every N events.
    """
    stats = ReplayStats()
    started_at = now()
    first_dt: int | None = None

    for event in events:
        if max_events and stats.events_sent >= max_events:
            break

        if first_dt is None:
            first_dt = event.transaction_dt
            stats.first_event_time = event.event_time

        if speedup > 0:
            # Schedule against the start of the run rather than the previous
            # event, so a slow send does not push the whole replay late.
            target = started_at + (event.transaction_dt - first_dt) / speedup
            delay = target - now()
            if delay >= MIN_SLEEP_SECONDS:
                sleep(delay)

        sink.send(event.card_key, event.model_dump_json())
        stats.events_sent += 1
        stats.last_event_time = event.event_time

        if progress_every and stats.events_sent % progress_every == 0:
            logger.info("%s events sent (now at %s)", stats.events_sent, event.event_time)

    sink.flush()
    stats.wall_seconds = now() - started_at
    return stats


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--transactions", type=Path, help="path to train_transaction.csv")
    parser.add_argument("--identity", type=Path, help="path to train_identity.csv")
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="force the synthetic fixture even if data/raw is present",
    )
    parser.add_argument("--topic", default=settings.kafka_topic_payments)
    parser.add_argument(
        "--bootstrap-servers",
        default=settings.kafka_bootstrap_servers_host,
        help="Kafka bootstrap servers (default: the host listener)",
    )
    parser.add_argument(
        "--speedup",
        type=float,
        default=settings.replay_speedup,
        help="dataset seconds per wall-clock second; 0 = as fast as possible",
    )
    parser.add_argument(
        "--limit", type=int, default=settings.producer_max_events, help="stop after N events"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the first few events instead of publishing to Kafka",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    settings = get_settings()

    if args.transactions:
        transactions_path, identity_path = args.transactions, args.identity
    else:
        transactions_path, identity_path, is_real = resolve_source_paths(
            prefer_raw=not args.fixture
        )
        if not is_real:
            logger.warning(
                "Using the synthetic fixture (%s). Run `make data` for the real dataset.",
                transactions_path.name,
            )

    transactions, identity_index = load_source(transactions_path, identity_path)
    logger.info(
        "loaded %s transactions and %s identity records from %s",
        len(transactions),
        len(identity_index),
        transactions_path,
    )

    epoch = parse_epoch(settings.synthetic_epoch)
    events = iter_payment_events(transactions, identity_index, epoch=epoch)

    if args.dry_run:
        sink: EventSink = ListSink()
        stats = replay(events, sink, speedup=0, max_events=args.limit or 5)
        for key, value in sink.messages[:5]:  # type: ignore[attr-defined]
            print(f"key={key}")
            print(json.dumps(json.loads(value), indent=2)[:800])
            print("-" * 60)
    else:
        kafka_sink = KafkaSink(args.bootstrap_servers, args.topic)
        logger.info(
            "publishing to topic %r at %s (speedup=%s)",
            args.topic,
            args.bootstrap_servers,
            args.speedup,
        )
        stats = replay(events, kafka_sink, speedup=args.speedup, max_events=args.limit)
        if kafka_sink.delivery_failures:
            logger.error("%s events failed to deliver", kafka_sink.delivery_failures)
            print(stats.describe())
            return 1

    print(stats.describe())
    return 0


if __name__ == "__main__":
    sys.exit(main())
