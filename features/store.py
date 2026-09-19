"""The online feature store: per-card state in Redis.

Read on every scoring request, so the access pattern is one point lookup by
card token. Redis does that in well under a millisecond, which is what the
~100 ms end-to-end budget requires.

Three properties worth knowing:

* **Keyed by token, not by card.** A dump of this Redis instance contains no
  card numbers, addresses or email domains (see ``common/pii.py``).
* **It is a cache, not a system of record.** The Delta Gold tables are the
  truth. State carries a TTL, so a card that stops transacting disappears
  rather than being scored against two-week-old history.
* **It stores state, not features.** The features themselves depend on the
  payment being scored (a 10-minute count includes the payment arriving now),
  so what is cached is the card's history and the features are computed on top
  of it by ``features/definitions.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from common.pii import card_token
from features.definitions import CardState

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable

KEY_PREFIX = "fp:card"


def state_key(token: str) -> str:
    """Redis key for one card's state."""
    return f"{KEY_PREFIX}:{token}"


class FeatureStore:
    """Read and write card state in Redis.

    The Redis client is injected, which is what lets every test here run
    against fakeredis with no server.
    """

    def __init__(self, client: Any, salt: str, ttl_seconds: int = 172_800) -> None:
        self._client = client
        self._salt = salt
        self._ttl = ttl_seconds

    def token_for(self, card_key: str) -> str:
        """Tokenise a raw card key with this store's salt."""
        return card_token(card_key, self._salt)

    def get_state(self, card_key: str) -> CardState:
        """Fetch a card's state. An unknown card returns empty state."""
        payload = self._client.get(state_key(self.token_for(card_key)))
        return CardState.from_json(payload)

    def get_state_by_token(self, token: str) -> CardState:
        return CardState.from_json(self._client.get(state_key(token)))

    def set_state(self, card_key: str, state: CardState) -> None:
        """Store a card's state with the configured TTL."""
        self._client.set(state_key(self.token_for(card_key)), state.to_json(), ex=self._ttl)

    def set_many(self, states: Iterable[tuple[str, CardState]]) -> int:
        """Write several cards in one pipeline round trip.

        The streaming job updates hundreds of cards per micro-batch; doing that
        as individual SETs would spend most of the batch waiting on the network.
        """
        pipeline = self._client.pipeline()
        written = 0
        for card_key, state in states:
            pipeline.set(state_key(self.token_for(card_key)), state.to_json(), ex=self._ttl)
            written += 1
        if written:
            pipeline.execute()
        return written

    def card_count(self) -> int:
        """How many cards currently have state. Diagnostics only - SCANs."""
        return sum(1 for _ in self._client.scan_iter(match=f"{KEY_PREFIX}:*"))


def build_redis_client(host: str, port: int, **kwargs: Any) -> Any:
    """Create a Redis client.

    Imported lazily so that this module can be used with fakeredis (and in
    environments with no redis package) without paying for the import.
    """
    import redis

    return redis.Redis(host=host, port=port, decode_responses=True, **kwargs)
