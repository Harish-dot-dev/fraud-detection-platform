"""Tests for the scoring path.

A real model, a real rules engine and a real (fake) Redis - only the
infrastructure is substituted. The decisions these tests assert on are the ones
a customer would experience.
"""

from __future__ import annotations

import pytest

from features.definitions import CardState, update_state
from features.store import FeatureStore
from serving.decisions import ALLOW, BLOCK, REVIEW, TRIGGER_MODEL, TRIGGER_RULE, ListDecisionSink
from serving.model import ModelBundle, degraded_bundle
from serving.rules import RulesEngine
from serving.scoring import ScoringService, payment_row
from training.preprocessing import MODEL_COLUMNS, build_matrix

SALT = "7" * 64


@pytest.fixture(scope="module")
def bundle() -> ModelBundle:
    from tests.demo_server import build_demo_bundle

    return build_demo_bundle(rows=3000)


@pytest.fixture
def store():
    import fakeredis

    return FeatureStore(fakeredis.FakeStrictRedis(decode_responses=True), salt=SALT)


@pytest.fixture
def service(store, bundle):
    return ScoringService(
        store=store, rules=RulesEngine.from_file(), bundle=bundle, sink=ListDecisionSink()
    )


def _warm(store, make_event, payments: int = 8) -> None:
    """Give the card a history of ordinary payments.

    A card's very first payment is genuinely unusual, so a test about
    *ordinary* traffic has to establish what ordinary looks like first.
    """
    event = make_event()
    state = CardState()
    for _ in range(payments):
        state = update_state(state, make_event(amount=40.0))
    store.set_state(event.card_key, state)


def test_an_ordinary_payment_is_allowed(store, service, make_event) -> None:
    _warm(store, make_event)

    decision = service.score(make_event(amount=42.0))

    assert decision.decision == ALLOW
    assert decision.score is not None
    assert decision.triggered_by == TRIGGER_MODEL
    assert decision.degraded is False


def test_a_blocklisted_card_is_blocked_without_running_the_model(store, bundle, make_event) -> None:
    """A confirmed-compromised card does not need a model's opinion."""
    event = make_event(amount=10.0)
    token = store.token_for(event.card_key)
    rules = RulesEngine.from_config(
        {
            "blocklist": {"card_tokens": [token]},
            "rules": [
                {
                    "name": "card_on_blocklist",
                    "description": "confirmed compromised",
                    "action": "block",
                    "when": {"all": [{"field": "card_token", "op": "in_blocklist"}]},
                }
            ],
        }
    )
    service = ScoringService(store=store, rules=rules, bundle=bundle, sink=ListDecisionSink())

    decision = service.score(event)

    assert decision.decision == BLOCK
    assert decision.triggered_by == TRIGGER_RULE
    assert decision.rule_name == "card_on_blocklist"
    # The model was skipped entirely, which is the latency win.
    assert decision.score is None


def test_a_review_rule_lifts_an_allow(service, make_event) -> None:
    """A policy limit sends a payment to an analyst whatever the model thinks."""
    decision = service.score(make_event(amount=9_000.0))

    assert decision.decision in {REVIEW, BLOCK}
    if decision.decision == REVIEW:
        assert decision.rule_name == "amount_over_hard_limit"


def test_the_model_can_still_block_when_a_review_rule_fires(store, bundle, make_event) -> None:
    """A review rule is a floor, not a veto."""
    rules = RulesEngine.from_config(
        {
            "rules": [
                {
                    "name": "always_review",
                    "description": "test rule",
                    "action": "review",
                    "when": {"all": [{"field": "amount", "op": ">", "value": 0}]},
                }
            ]
        }
    )
    # A bundle that blocks everything.
    blocking = ModelBundle(
        model=bundle.model,
        version=bundle.version,
        review_threshold=0.0,
        block_threshold=0.0,
        source="test",
    )
    service = ScoringService(store=store, rules=rules, bundle=blocking, sink=ListDecisionSink())

    decision = service.score(make_event(amount=50.0))

    assert decision.decision == BLOCK
    assert decision.triggered_by == TRIGGER_MODEL


def test_the_card_history_is_used(store, service, make_event) -> None:
    """The whole point of the online feature store."""
    event = make_event(amount=100.0)
    state = CardState()
    for _ in range(6):
        state = update_state(state, make_event(amount=100.0))
    store.set_state(event.card_key, state)

    decision = service.score(event)

    assert decision.features["card_txn_count_10m"] == 7.0
    assert decision.features["card_is_new"] == 0.0


def test_scoring_never_writes_to_the_feature_store(store, service, make_event) -> None:
    """The streaming job owns that state.

    If the API updated it too, a retried request would count the same payment
    twice in its own card's history.
    """
    event = make_event()

    service.score(event)
    service.score(event)

    assert store.get_state(event.card_key).is_empty


def test_a_broken_feature_store_does_not_fail_the_payment(bundle, make_event) -> None:
    """Redis is a cache. Losing it must degrade the score, not the service."""

    class BrokenStore(FeatureStore):
        def get_state(self, card_key: str):
            raise ConnectionError("redis is down")

    import fakeredis

    store = BrokenStore(fakeredis.FakeStrictRedis(decode_responses=True), salt=SALT)
    service = ScoringService(
        store=store, rules=RulesEngine.from_file(), bundle=bundle, sink=ListDecisionSink()
    )

    decision = service.score(make_event(amount=25.0))

    # Scored as an unfamiliar card, which biases towards caution.
    assert decision.features["card_is_new"] == 1.0
    assert decision.decision in {ALLOW, REVIEW, BLOCK}


def test_without_a_model_the_rules_still_run(store, make_event) -> None:
    """Degraded mode: a platform that refuses to answer is worse than a cautious one."""
    service = ScoringService(
        store=store,
        rules=RulesEngine.from_file(),
        bundle=degraded_bundle(0.3, 0.8),
        sink=ListDecisionSink(),
    )

    ordinary = service.score(make_event(amount=20.0))
    large = service.score(make_event(amount=9_000.0))

    assert ordinary.decision == ALLOW
    assert ordinary.degraded is True
    assert ordinary.score is None
    assert large.decision == REVIEW
    assert large.rule_name == "amount_over_hard_limit"


def test_flagged_payments_get_reasons_and_allowed_ones_do_not(store, service, make_event) -> None:
    """SHAP runs only where somebody will read it - it is not free."""
    _warm(store, make_event)

    allowed = service.score(make_event(amount=30.0))
    flagged = service.score(make_event(amount=9_000.0))

    assert allowed.top_reasons == []
    assert flagged.top_reasons
    assert all(reason["contribution"] > 0 for reason in flagged.top_reasons)
    assert flagged.top_reasons[0]["feature"] in MODEL_COLUMNS


def test_every_decision_is_auditable(service, make_event) -> None:
    """Requirement: reproduce any decision months later."""
    decision = service.score(make_event(amount=120.0))

    assert decision.transaction_id
    assert decision.card_token and "gmail.com" not in decision.card_token
    assert decision.decided_at
    assert decision.model_version
    assert decision.features
    assert decision.latency_ms > 0
    assert decision.reason
    # And it serialises for the topic.
    assert '"decision"' in decision.to_json()


def test_decisions_reach_the_sink(store, bundle, make_event) -> None:
    sink = ListDecisionSink()
    service = ScoringService(store=store, rules=RulesEngine.from_file(), bundle=bundle, sink=sink)

    service.score(make_event())
    service.score(make_event())

    assert len(sink.decisions) == 2


def test_the_scoring_row_matches_the_training_schema(make_event) -> None:
    """Serving builds the model's input from an event; training builds it from
    the Gold table. Both must produce the same matrix."""
    from features.definitions import compute_features

    event = make_event(amount=75.0, device_info="Windows", r_emaildomain="gmail.com")
    row = payment_row(event, compute_features(CardState(), event))

    import pandas as pd

    matrix = build_matrix(pd.DataFrame([row]))

    assert list(matrix.columns) == MODEL_COLUMNS
    assert matrix["amount"].iloc[0] == 75.0
    assert len(matrix) == 1
