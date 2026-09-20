"""The audit record written for every payment the platform scores.

A fraud decision affects a customer, so it has to be explainable months later,
to an analyst, a complaints team or a regulator. That means recording not just
what was decided but everything the decision was made from: the model version,
the feature values it saw, which rule or threshold triggered, the reasons, and
how long it took.

Decisions are published to Kafka rather than written to Delta inline. A
synchronous table write on the scoring path would put file-system latency
between a customer and their payment; the topic is durable, and phase 7's
consumer lands it in Delta for analysis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger("serving.decisions")

ALLOW = "allow"
REVIEW = "review"
BLOCK = "block"

TRIGGER_RULE = "rule"
TRIGGER_MODEL = "model"
TRIGGER_DEFAULT = "default"


@dataclass
class Decision:
    """One scoring decision, in full."""

    transaction_id: int
    card_token: str
    decided_at: datetime
    decision: str
    # None when a rule blocked the payment before the model ran.
    score: float | None
    # "rule", "model" or "default" - what actually made this call.
    triggered_by: str
    reason: str
    rule_name: str | None = None
    matched_rules: list[str] = field(default_factory=list)
    model_version: str = "none"
    review_threshold: float = 0.0
    block_threshold: float = 1.0
    top_reasons: list[dict[str, Any]] = field(default_factory=list)
    # The feature values the decision was made from, so it can be reproduced.
    features: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    # True when the model could not be loaded and only rules ran.
    degraded: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["decided_at"] = self.decided_at.isoformat()
        # A SHAP reason's value is a number for a numeric feature and a label
        # for a categorical one. Left as a union it would be unreadable by the
        # Spark schema that lands this topic in Delta, so it is always a
        # string on the wire; the typed value stays available in Python.
        payload["top_reasons"] = [
            {**reason, "value": None if reason.get("value") is None else str(reason["value"])}
            for reason in payload.get("top_reasons") or []
        ]
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), default=str)


def now() -> datetime:
    return datetime.now(UTC)


class DecisionSink(Protocol):
    """Anywhere decisions are recorded."""

    def record(self, decision: Decision) -> None: ...

    def close(self) -> None: ...


class ListDecisionSink:
    """Keeps decisions in memory. Used by the tests and the load test."""

    def __init__(self) -> None:
        self.decisions: list[Decision] = []

    def record(self, decision: Decision) -> None:
        self.decisions.append(decision)

    def close(self) -> None:
        return None


class JsonlDecisionSink:
    """Appends decisions to a JSON Lines file.

    The fallback when Kafka is not running - a demo on a laptop still leaves a
    complete audit trail behind.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, decision: Decision) -> None:
        with self.path.open("a") as handle:
            handle.write(decision.to_json() + "\n")

    def close(self) -> None:
        return None


class KafkaDecisionSink:
    """Publishes decisions to the decisions topic.

    Deliberately fire-and-forget: ``produce`` hands the message to the client's
    background thread and returns. A payment must not wait on a broker
    acknowledgement, and a failed delivery is logged rather than raised - the
    customer's payment has already been decided either way.
    """

    def __init__(self, bootstrap_servers: str, topic: str) -> None:
        from confluent_kafka import Producer

        self.topic = topic
        self.delivery_failures = 0
        self._producer = Producer(
            {"bootstrap.servers": bootstrap_servers, "linger.ms": 20, "compression.type": "lz4"}
        )

    def _on_delivery(self, error: Any, _message: Any) -> None:
        if error is not None:
            self.delivery_failures += 1
            logger.error("decision delivery failed: %s", error)

    def record(self, decision: Decision) -> None:
        try:
            self._producer.produce(
                self.topic,
                key=str(decision.transaction_id).encode(),
                value=decision.to_json().encode(),
                on_delivery=self._on_delivery,
            )
            self._producer.poll(0)
        except BufferError:
            # The local queue is full: drop the record rather than block the
            # scoring path, and make the loss visible.
            self.delivery_failures += 1
            logger.error("decision queue full; dropped decision for %s", decision.transaction_id)

    def close(self) -> None:
        self._producer.flush(10)
