"""The scoring path: features, rules, model, reasons - in that order.

This is what a payment actually goes through, and every step is arranged
around the latency budget:

1. **Read the card's state from Redis.** One point lookup. If Redis is
   unreachable the card is treated as unknown rather than the request failing:
   an unfamiliar card scores conservatively, which is the safe direction.
2. **Compute the features**, including this payment, using the shared
   definitions - the same function the batch path uses.
3. **Run the rules.** A blocking rule short-circuits: the model is not run at
   all. It saves several milliseconds, and the decision was not the model's to
   make anyway.
4. **Score with the model**, and turn the probability into allow / review /
   block with the thresholds that were tuned for this model version.
5. **Explain, but only when it matters.** SHAP runs for review and block
   decisions, not for the ~97% of payments that are allowed. Nobody reads the
   reasons for an allowed payment, and it keeps the common path fast.
6. **Record the decision** to the audit sink, and return.

The service never writes to Redis. The streaming job owns that state, so a
retried request cannot double-count a payment in its own card's history.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

from common.events import PaymentEvent
from features.definitions import compute_features
from features.store import FeatureStore
from serving.decisions import (
    ALLOW,
    BLOCK,
    REVIEW,
    TRIGGER_DEFAULT,
    TRIGGER_MODEL,
    TRIGGER_RULE,
    Decision,
    DecisionSink,
    ListDecisionSink,
    now,
)
from serving.model import ModelBundle
from serving.rules import RulesEngine
from training.preprocessing import build_row

logger = logging.getLogger("serving.scoring")

# Payment attributes the rules can test, alongside the computed features.
EVENT_FIELDS = (
    "product_cd",
    "card3",
    "card4",
    "card5",
    "card6",
    "dist1",
    "dist2",
    "p_emaildomain",
    "r_emaildomain",
    "device_type",
    "device_info",
)


def payment_row(event: PaymentEvent, features: dict[str, float]) -> dict[str, Any]:
    """Assemble the Gold-shaped row the model expects for one payment.

    Training reads these columns out of the Gold table; here they come straight
    off the event. Same names, same meanings - ``build_matrix`` then produces
    an identical matrix either way.
    """
    return {
        **features,
        **{name: getattr(event, name, None) for name in EVENT_FIELDS},
        "counts": event.counts,
        "deltas": event.deltas,
    }


class ScoringService:
    """Turns a payment into a decision."""

    def __init__(
        self,
        store: FeatureStore,
        rules: RulesEngine,
        bundle: ModelBundle,
        sink: DecisionSink | None = None,
        explain_top_n: int = 3,
    ) -> None:
        self.store = store
        self.rules = rules
        self.bundle = bundle
        self.sink = sink or ListDecisionSink()
        self.explain_top_n = explain_top_n

    def score(self, event: PaymentEvent) -> Decision:
        """Score one payment and record the decision."""
        started = time.perf_counter()

        features = compute_features(self._card_state(event), event)
        card_token = self.store.token_for(event.card_key)

        context = {**payment_row(event, features), "card_token": card_token}
        rule_outcome = self.rules.evaluate(context)

        score: float | None = None
        matrix: pd.DataFrame | None = None
        decision = ALLOW
        triggered_by = TRIGGER_DEFAULT
        reason = "no rule matched and the score was below the review threshold"

        if rule_outcome.blocks:
            # Short-circuit: a confirmed-compromised card does not need a model.
            decision = BLOCK
            triggered_by = TRIGGER_RULE
            reason = rule_outcome.description or f"rule {rule_outcome.rule_name}"
        else:
            if self.bundle.is_loaded:
                matrix = build_row(payment_row(event, features))
                score = self.bundle.predict(matrix)
                decision, triggered_by, reason = self._decide(score)

            if rule_outcome.action == REVIEW and decision == ALLOW:
                # A review rule is a floor, not a veto: it cannot override the
                # model's decision to block, but it can lift an allow.
                decision = REVIEW
                triggered_by = TRIGGER_RULE
                reason = rule_outcome.description or f"rule {rule_outcome.rule_name}"

        top_reasons = self._explain(matrix) if decision != ALLOW else []

        record = Decision(
            transaction_id=event.transaction_id,
            card_token=card_token,
            decided_at=now(),
            decision=decision,
            score=score,
            triggered_by=triggered_by,
            reason=reason,
            rule_name=rule_outcome.rule_name,
            matched_rules=rule_outcome.matched,
            model_version=self.bundle.version,
            review_threshold=self.bundle.review_threshold,
            block_threshold=self.bundle.block_threshold,
            top_reasons=top_reasons,
            features=features,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            degraded=not self.bundle.is_loaded,
        )
        self.sink.record(record)
        return record

    def _card_state(self, event: PaymentEvent):
        """The card's history, or an empty history if Redis is unavailable."""
        from features.definitions import CardState

        try:
            return self.store.get_state(event.card_key)
        except Exception as error:  # noqa: BLE001 - never fail a payment on a cache
            logger.warning("feature store unavailable (%s); scoring as an unknown card", error)
            return CardState()

    def _decide(self, score: float) -> tuple[str, str, str]:
        if score >= self.bundle.block_threshold:
            return (
                BLOCK,
                TRIGGER_MODEL,
                f"model score {score:.3f} at or above the block threshold "
                f"{self.bundle.block_threshold:.3f}",
            )
        if score >= self.bundle.review_threshold:
            return (
                REVIEW,
                TRIGGER_MODEL,
                f"model score {score:.3f} at or above the review threshold "
                f"{self.bundle.review_threshold:.3f}",
            )
        return (
            ALLOW,
            TRIGGER_MODEL,
            f"model score {score:.3f} below the review threshold "
            f"{self.bundle.review_threshold:.3f}",
        )

    def _explain(self, matrix: pd.DataFrame | None) -> list[dict[str, Any]]:
        """Top SHAP reasons, when there is a model and a row to explain."""
        if matrix is None or self.bundle.explainer is None:
            return []
        try:
            reasons = self.bundle.explainer.top_reasons(matrix, top_n=self.explain_top_n)
            return [reason.to_dict() for reason in reasons]
        except Exception as error:  # noqa: BLE001 - reasons are not worth a 500
            logger.warning("could not explain the decision (%s)", error)
            return []
