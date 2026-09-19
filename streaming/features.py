"""The online feature pipeline: Kafka -> Bronze Delta + Redis.

This is the job the ``spark`` container runs. For every micro-batch it does two
things with one read of Kafka:

1. appends the raw events to the Bronze Delta table, and
2. folds them into each card's state in Redis, which is what the scoring API
   reads on the next payment.

Doing both in one ``foreachBatch`` is a deliberate concession to running on a
laptop: two separate jobs would mean two Spark drivers, two JVM heaps and two
reads of the same topic. On a real cluster these would be separate
applications with separate checkpoints, so that a problem in the feature
writer could not stall ingestion.

Ordering matters here. State updates are not commutative - "was this device
seen before?" depends on what came earlier - so each micro-batch is
repartitioned by card and sorted within the partition before the fold.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from common.config import get_settings
from common.events import PaymentEvent
from common.spark import KAFKA_PACKAGE, build_spark_session
from features.definitions import CardState, update_state
from features.store import FeatureStore, build_redis_client

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame

logger = logging.getLogger("online-features")


@dataclass(frozen=True)
class RedisConfig:
    """Connection details shipped to the Spark workers.

    Plain strings and ints only: this object is pickled and sent to every
    executor, so it must not hold a client, a session or anything else live.
    """

    host: str
    port: int
    salt: str
    ttl_seconds: int


def row_to_event(row: Any) -> PaymentEvent:
    """Convert a Bronze row back into the event the feature code expects.

    Only the fields the state update actually reads are populated; the rest of
    the payload is already safely in Bronze.
    """
    return PaymentEvent(
        transaction_id=row["transaction_id"],
        event_time=row["event_time"],
        transaction_dt=row["transaction_dt"],
        card_key=row["card_key"],
        amount=row["amount"],
        device_info=row["device_info"],
        r_emaildomain=row["r_emaildomain"],
    )


def update_states(events: Iterable[PaymentEvent], store: FeatureStore) -> int:
    """Fold a batch of events into card state and write it back.

    Args:
        events: payments for this partition, **in time order per card**.
        store: the online feature store.

    Returns:
        the number of cards written.

    State for a card is read from Redis once and then carried in memory for the
    rest of the batch: a card with twenty payments in one micro-batch costs one
    read and one write, not twenty of each.
    """
    states: dict[str, CardState] = {}

    for event in events:
        if event.card_key not in states:
            states[event.card_key] = store.get_state(event.card_key)
        states[event.card_key] = update_state(states[event.card_key], event)

    return store.set_many(states.items())


def _update_partition(rows: Iterator[Any], config: RedisConfig) -> None:
    """Executor-side entry point: one Redis connection per partition."""
    store = FeatureStore(
        build_redis_client(config.host, config.port),
        salt=config.salt,
        ttl_seconds=config.ttl_seconds,
    )
    update_states((row_to_event(row) for row in rows), store)


def process_batch(events: DataFrame, bronze_path: str, config: RedisConfig) -> None:
    """Handle one micro-batch: Bronze first, then the online store.

    Bronze is written first on purpose. If the Redis update fails, the raw
    events are already durable and the state can be rebuilt from them; the
    other order would risk losing the payment entirely.
    """
    from streaming.bronze import write_bronze_batch

    # Spark can recompute a cached batch otherwise, re-reading Kafka.
    events.persist()
    try:
        write_bronze_batch(events, bronze_path)

        (
            events.select(
                "transaction_id",
                "event_time",
                "transaction_dt",
                "card_key",
                "amount",
                "device_info",
                "r_emaildomain",
            )
            # One partition per card keeps a card's payments together, and the
            # sort puts them in the order they happened.
            .repartition("card_key")
            .sortWithinPartitions("card_key", "transaction_dt")
            .foreachPartition(lambda rows: _update_partition(rows, config))
        )
    finally:
        events.unpersist()


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bootstrap-servers", default=settings.kafka_bootstrap_servers)
    parser.add_argument("--topic", default=settings.kafka_topic_payments)
    parser.add_argument("--bronze-path", default=str(settings.path(settings.delta_bronze_path)))
    parser.add_argument(
        "--checkpoint-path",
        default=str(settings.path(settings.delta_bronze_path) / "_checkpoints" / "online"),
    )
    parser.add_argument("--starting-offsets", default="earliest", choices=["earliest", "latest"])
    parser.add_argument("--trigger-interval", default="10 seconds")
    parser.add_argument("--once", action="store_true", help="drain the topic and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    settings = get_settings()

    from streaming.bronze import parse_payment_events, read_payments_stream

    config = RedisConfig(
        host=settings.redis_host,
        port=settings.redis_port,
        salt=settings.pii_hash_salt,
        ttl_seconds=settings.redis_feature_ttl_seconds,
    )

    spark = build_spark_session("online-features", packages=[KAFKA_PACKAGE])
    spark.sparkContext.setLogLevel("WARN")

    events = parse_payment_events(
        read_payments_stream(
            spark, args.bootstrap_servers, args.topic, starting_offsets=args.starting_offsets
        )
    )

    writer = (
        events.writeStream.foreachBatch(
            lambda batch, _batch_id: process_batch(batch, args.bronze_path, config)
        )
        .option("checkpointLocation", args.checkpoint_path)
        .outputMode("append")
    )
    writer = (
        writer.trigger(availableNow=True)
        if args.once
        else writer.trigger(processingTime=args.trigger_interval)
    )

    logger.info(
        "bronze -> %s, online features -> redis://%s:%s",
        args.bronze_path,
        config.host,
        config.port,
    )
    query = writer.start()
    query.awaitTermination()
    return 0


if __name__ == "__main__":
    sys.exit(main())
