"""The single definition of every behavioural feature in the platform.

This module is the answer to the question that sinks most real feature
pipelines: *why does the model see different numbers in production than it saw
in training?*

Usually it is because the streaming job and the training job each implement
"transactions on this card in the last hour" separately, and the two drift.
Here both paths call the functions below. The online path (Spark -> Redis ->
the scoring API) and the offline path (Delta Gold, for training) therefore
compute the same number by construction, and
``tests/test_feature_consistency.py`` proves it against a second,
window-function implementation used for batch efficiency.

The shape of the API is deliberate::

    features = compute_features(state_before, event)
    state_after = update_state(state_before, event)

``state_before`` holds only what happened *strictly before* this payment, so
every feature is point-in-time correct by construction: there is no way to see
the future because the future is not in the state object.

Features that describe the current payment (its amount, its hour of day) do
use the event itself - that is information the system genuinely has at
authorisation time.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from common.events import PaymentEvent

# --- Windows ---------------------------------------------------------------
# Three horizons: a burst of payments in ten minutes looks very different from
# steady activity over a day, and the model gets to see both.
WINDOW_10M = timedelta(minutes=10)
WINDOW_1H = timedelta(hours=1)
WINDOW_24H = timedelta(hours=24)
LONGEST_WINDOW = WINDOW_24H

# Bounded so that a single very active card cannot grow its Redis entry without
# limit. Both numbers are generous relative to real card behaviour.
MAX_TRACKED_DEVICES = 20
MAX_TRACKED_EMAIL_DOMAINS = 20

# Returned when a feature has no history to be computed from. -1 rather than 0
# or NaN: 0 would mean "no time has passed", and XGBoost handles a distinctive
# sentinel perfectly well while NaN would hide the distinction between "first
# payment on this card" and "value genuinely missing".
NO_HISTORY = -1.0

# The feature vector the model is trained and scored on, in a fixed order.
# Anything that changes this list must retrain the model.
FEATURE_NAMES: list[str] = [
    # Stateless: derived from the payment itself.
    "amount",
    "amount_log",
    "hour_of_day",
    "day_of_week",
    "is_night",
    "has_identity",
    # Velocity: how busy this card has been.
    "card_txn_count_10m",
    "card_txn_count_1h",
    "card_txn_count_24h",
    "card_amount_sum_24h",
    "seconds_since_card_last_txn",
    # Spending pattern: is this payment normal for this card?
    "card_amount_avg_lifetime",
    "amount_to_card_avg_ratio",
    "card_txn_count_lifetime",
    # Novelty: things this card has not done before.
    "card_is_new",
    "card_new_device",
    "card_new_email_domain",
    "card_distinct_devices",
]


@dataclass
class CardState:
    """Everything the platform remembers about one card.

    This is exactly what lives in Redis under the card's key, and exactly what
    the offline reference implementation carries forward row by row. Keeping
    one structure for both is what makes the two paths comparable.

    ``recent_events`` holds ``(epoch_seconds, amount)`` pairs inside the longest
    window only - the windowed counts need the individual timestamps, but
    nothing needs them once they age out.
    """

    txn_count_lifetime: int = 0
    amount_sum_lifetime: float = 0.0
    last_event_epoch: float | None = None
    recent_events: list[tuple[float, float]] = field(default_factory=list)
    known_devices: list[str] = field(default_factory=list)
    known_email_domains: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True when this card has never been seen before."""
        return self.txn_count_lifetime == 0

    @property
    def amount_avg_lifetime(self) -> float:
        if self.txn_count_lifetime == 0:
            return 0.0
        return self.amount_sum_lifetime / self.txn_count_lifetime

    def to_json(self) -> str:
        """Serialise for Redis. Compact keys keep the online store small."""
        return json.dumps(
            {
                "n": self.txn_count_lifetime,
                "sum": round(self.amount_sum_lifetime, 4),
                "last": self.last_event_epoch,
                "recent": [[round(ts, 3), round(amount, 4)] for ts, amount in self.recent_events],
                "devices": self.known_devices,
                "domains": self.known_email_domains,
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, payload: str | bytes | None) -> CardState:
        """Deserialise from Redis. A missing or corrupt entry means "new card".

        Treating a corrupt entry as a new card is the safe failure mode: the
        payment still gets scored, it simply looks unfamiliar, which biases
        towards review rather than towards a silent allow.
        """
        if not payload:
            return cls()
        try:
            raw = json.loads(payload)
        except (ValueError, TypeError):
            return cls()

        return cls(
            txn_count_lifetime=int(raw.get("n", 0)),
            amount_sum_lifetime=float(raw.get("sum", 0.0)),
            last_event_epoch=raw.get("last"),
            recent_events=[(float(ts), float(amount)) for ts, amount in raw.get("recent", [])],
            known_devices=list(raw.get("devices", [])),
            known_email_domains=list(raw.get("domains", [])),
        )


def _epoch_seconds(moment: datetime) -> float:
    return moment.timestamp()


def _count_and_sum_within(
    recent_events: list[tuple[float, float]], now_epoch: float, window: timedelta
) -> tuple[int, float]:
    """Count events and total their amounts inside a trailing window.

    The window is half-open: ``(now - window, now]``. An event exactly on the
    boundary counts as inside, which keeps the online and offline paths from
    disagreeing on an edge case that is otherwise easy to get wrong.

    Both ends are bounded. In the online path state only ever holds payments
    that already happened, so an upper bound looks redundant - but it stops a
    late-arriving payment, folded in after a newer one, from counting events
    from its own future. That would be a leak, and it is exactly what the
    offline ``rangeBetween(-window, currentRow)`` refuses to do.
    """
    cutoff = now_epoch - window.total_seconds()
    count = 0
    total = 0.0
    for timestamp, amount in recent_events:
        if cutoff <= timestamp <= now_epoch:
            count += 1
            total += amount
    return count, total


def stateless_features(event: PaymentEvent) -> dict[str, float]:
    """Features that need no history - available from the payment alone."""
    hour = event.event_time.hour
    return {
        "amount": event.amount,
        # Payment amounts are heavily skewed; the log is what the model can
        # actually use for a split.
        "amount_log": math.log1p(max(event.amount, 0.0)),
        "hour_of_day": float(hour),
        "day_of_week": float(event.event_time.weekday()),
        # Card-not-present fraud clusters at night in the cardholder's timezone.
        # This is a crude proxy - the dataset has no timezone - and is
        # documented as such.
        "is_night": float(hour < 6),
        "has_identity": float(bool(event.device_info or event.identity_numeric)),
    }


def compute_features(state: CardState, event: PaymentEvent) -> dict[str, float]:
    """Compute the full feature vector for ``event`` given the card's history.

    Args:
        state: what the card had done *before* this payment.
        event: the payment being scored.

    Returns:
        A dict keyed by :data:`FEATURE_NAMES`.

    The windowed counts include the current payment: at scoring time "three
    payments in the last ten minutes" naturally means this one and two before
    it. That is also why the scoring API cannot simply read a precomputed value
    out of Redis - the value it needs does not exist until the payment arrives.
    """
    now_epoch = _epoch_seconds(event.event_time)

    count_10m, _ = _count_and_sum_within(state.recent_events, now_epoch, WINDOW_10M)
    count_1h, _ = _count_and_sum_within(state.recent_events, now_epoch, WINDOW_1H)
    count_24h, amount_24h = _count_and_sum_within(state.recent_events, now_epoch, WINDOW_24H)

    if state.last_event_epoch is None:
        seconds_since_last = NO_HISTORY
    else:
        seconds_since_last = max(now_epoch - state.last_event_epoch, 0.0)

    average = state.amount_avg_lifetime
    # A card with no history has no baseline to compare against, so the ratio
    # is neutral rather than infinite.
    ratio = event.amount / average if average > 0 else 1.0

    device = event.device_info
    domain = event.r_emaildomain

    features = {
        # The current payment is part of its own window.
        "card_txn_count_10m": float(count_10m + 1),
        "card_txn_count_1h": float(count_1h + 1),
        "card_txn_count_24h": float(count_24h + 1),
        "card_amount_sum_24h": amount_24h + event.amount,
        "seconds_since_card_last_txn": seconds_since_last,
        "card_amount_avg_lifetime": average,
        "amount_to_card_avg_ratio": ratio,
        "card_txn_count_lifetime": float(state.txn_count_lifetime),
        "card_is_new": float(state.is_empty),
        # "New" only means something once the card has some history: on a
        # card's first payment every device is new, which is not a signal.
        "card_new_device": float(
            bool(device) and not state.is_empty and device not in state.known_devices
        ),
        "card_new_email_domain": float(
            bool(domain) and not state.is_empty and domain not in state.known_email_domains
        ),
        "card_distinct_devices": float(len(state.known_devices)),
    }
    features.update(stateless_features(event))
    return {name: features[name] for name in FEATURE_NAMES}


def update_state(state: CardState, event: PaymentEvent) -> CardState:
    """Fold a payment into the card's state, returning the new state.

    Pure: the input state is not mutated, so a caller can compute features from
    the old state and update afterwards without worrying about ordering.
    """
    now_epoch = _epoch_seconds(event.event_time)

    # Drop events that have aged out of the longest window, then add this one.
    cutoff = now_epoch - LONGEST_WINDOW.total_seconds()
    recent = [(ts, amount) for ts, amount in state.recent_events if ts >= cutoff]
    recent.append((now_epoch, event.amount))

    devices = list(state.known_devices)
    if event.device_info and event.device_info not in devices:
        devices.append(event.device_info)
        devices = devices[-MAX_TRACKED_DEVICES:]

    domains = list(state.known_email_domains)
    if event.r_emaildomain and event.r_emaildomain not in domains:
        domains.append(event.r_emaildomain)
        domains = domains[-MAX_TRACKED_EMAIL_DOMAINS:]

    return CardState(
        txn_count_lifetime=state.txn_count_lifetime + 1,
        amount_sum_lifetime=state.amount_sum_lifetime + event.amount,
        # max() guards against an out-of-order event moving the clock backwards.
        last_event_epoch=max(now_epoch, state.last_event_epoch or now_epoch),
        recent_events=recent,
        known_devices=devices,
        known_email_domains=domains,
    )


def compute_and_update(state: CardState, event: PaymentEvent) -> tuple[dict[str, float], CardState]:
    """Convenience for the common "score then remember" sequence."""
    return compute_features(state, event), update_state(state, event)
