"""The scoring consumer: payments topic -> /score.

Kept separate from the API on purpose. The API is a synchronous service with a
latency budget; this is a worker that can be scaled, paused or replayed
independently. It also means the same endpoint serves the consumer, the
analyst app and a curl command, so there is one scoring path rather than two
that drift.

    make consume                       # against the running stack
    make consume ARGS="--limit 100"

Offsets are committed only after a payment has been scored, so a crash
re-delivers the payment rather than losing it. Scoring is a read-only
operation on the feature store, so a re-delivery is harmless - it produces the
same decision twice rather than corrupting a card's history.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass

import httpx

from common.config import get_settings

logger = logging.getLogger("consumer")


@dataclass
class ConsumerStats:
    """What a run of the consumer did."""

    scored: int = 0
    failed: int = 0
    decisions: dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.decisions = self.decisions or {"allow": 0, "review": 0, "block": 0}

    def describe(self) -> str:
        mix = ", ".join(f"{count} {name}" for name, count in self.decisions.items())
        return f"scored {self.scored} payments ({mix}); {self.failed} failed"


def score_payload(client: httpx.Client, api_url: str, payload: str) -> dict | None:
    """Send one payment to the API and return its decision."""
    response = client.post(
        f"{api_url}/score", content=payload, headers={"content-type": "application/json"}
    )
    response.raise_for_status()
    return response.json()


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bootstrap-servers", default=settings.kafka_bootstrap_servers_host)
    parser.add_argument("--topic", default=settings.kafka_topic_payments)
    parser.add_argument("--api-url", default=settings.scoring_api_url_host)
    parser.add_argument("--group-id", default="scoring-consumer")
    parser.add_argument("--limit", type=int, default=0, help="stop after N payments (0 = forever)")
    parser.add_argument(
        "--from-beginning",
        action="store_true",
        help="read the topic from the start instead of from the last commit",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="seconds to wait for a message")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    from confluent_kafka import Consumer

    consumer = Consumer(
        {
            "bootstrap.servers": args.bootstrap_servers,
            "group.id": args.group_id,
            "auto.offset.reset": "earliest" if args.from_beginning else "latest",
            # Commit only once a payment has actually been scored.
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([args.topic])
    stats = ConsumerStats()

    logger.info("consuming %s -> %s", args.topic, args.api_url)
    try:
        with httpx.Client(timeout=10.0) as client:
            while not args.limit or stats.scored < args.limit:
                message = consumer.poll(args.timeout)
                if message is None:
                    logger.info("no messages for %.0fs; stopping", args.timeout)
                    break
                if message.error():
                    logger.error("kafka error: %s", message.error())
                    continue

                try:
                    decision = score_payload(client, args.api_url, message.value().decode())
                    stats.scored += 1
                    stats.decisions[decision["decision"]] += 1
                    if decision["decision"] != "allow":
                        logger.info(
                            "%s %s (score=%s, %s)",
                            decision["decision"].upper(),
                            decision["transaction_id"],
                            decision["score"],
                            decision["reason"],
                        )
                except (httpx.HTTPError, json.JSONDecodeError, KeyError) as error:
                    stats.failed += 1
                    logger.error("scoring failed: %s", error)

                consumer.commit(message=message, asynchronous=False)
    finally:
        consumer.close()

    print(stats.describe())
    return 0


if __name__ == "__main__":
    sys.exit(main())
