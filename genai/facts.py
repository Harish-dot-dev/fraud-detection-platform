"""The facts object: everything the LLM is allowed to know.

The single most important rule in this part of the system is that **the model
never computes a number**. Every figure an analyst reads - the amount, the
velocity count, the score, the multiple of the card's average - is calculated
by the pipeline, put in this object, and handed to the LLM to write prose
around.

A 3B model asked to work out "how many times the usual amount is 847.20 when
the average is 92.15" will produce a number. It will be wrong often enough to
matter, and wrong in a way that looks authoritative on an analyst's screen.

So the LLM's job is narrow: turn facts into readable English, and say when
something is not supported. ``supported_numbers`` is what makes that checkable
- every number in the generated summary has to appear here, and
``genai/summarise.py`` rejects the output if it does not.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from genai.case_store import PastCase
from serving.decisions import Decision

# Features worth putting in front of an analyst, with a plain-English label.
# The model sees 57 columns; a human needs the handful that carry the story.
NARRATIVE_FEATURES: dict[str, str] = {
    "amount": "payment amount",
    "card_txn_count_10m": "payments on this card in the last 10 minutes",
    "card_txn_count_24h": "payments on this card in the last 24 hours",
    "card_amount_sum_24h": "total charged to this card in 24 hours",
    "amount_to_card_avg_ratio": "multiple of this card's average payment",
    "card_amount_avg_lifetime": "this card's average payment",
    "seconds_since_card_last_txn": "seconds since the previous payment",
    "card_is_new": "first payment ever seen on this card (1 = yes)",
    "card_new_device": "device never seen on this card before (1 = yes)",
    "card_new_email_domain": "recipient email domain new for this card (1 = yes)",
    "is_night": "made between midnight and 6am (1 = yes)",
}


@dataclass
class CaseFacts:
    """A flagged payment, reduced to what can be stated as fact."""

    transaction_id: int
    card_token: str
    decided_at: datetime
    decision: str
    score: float | None
    triggered_by: str
    reason: str
    rule_name: str | None
    model_version: str
    review_threshold: float
    block_threshold: float
    amount: float
    features: dict[str, float] = field(default_factory=dict)
    top_reasons: list[dict[str, Any]] = field(default_factory=list)
    similar_cases: list[PastCase] = field(default_factory=list)

    @property
    def similar_case_ids(self) -> list[int]:
        return [case.transaction_id for case in self.similar_cases]

    def supported_numbers(self) -> set[float]:
        """Every number the summary is allowed to contain.

        Anything else is the model inventing arithmetic.
        """
        numbers: set[float] = {self.amount, self.review_threshold, self.block_threshold}
        if self.score is not None:
            numbers.add(self.score)
        numbers.update(float(value) for value in self.features.values())
        numbers.update(
            float(reason["contribution"])
            for reason in self.top_reasons
            if reason.get("contribution") is not None
        )
        for case in self.similar_cases:
            numbers.add(float(case.amount))
            numbers.add(float(case.transaction_id))
            if case.similarity is not None:
                numbers.add(float(case.similarity))
        numbers.add(float(len(self.similar_cases)))
        numbers.add(float(self.transaction_id))
        return numbers

    def to_prompt_block(self) -> str:
        """The facts, formatted for the prompt."""
        lines = [
            "PAYMENT UNDER REVIEW",
            f"- transaction id: {self.transaction_id}",
            f"- amount: {self.amount:.2f}",
            f"- decision: {self.decision}",
            f"- decided by: {self.triggered_by}"
            + (f" (rule: {self.rule_name})" if self.rule_name else ""),
        ]
        if self.score is not None:
            lines.append(
                f"- model score: {self.score:.4f} "
                f"(review at {self.review_threshold:.4f}, block at {self.block_threshold:.4f})"
            )

        lines.append("")
        lines.append("BEHAVIOURAL FACTS")
        for name, label in NARRATIVE_FEATURES.items():
            if name in self.features:
                lines.append(f"- {label}: {_format_number(self.features[name])}")

        if self.top_reasons:
            lines.append("")
            lines.append("WHAT DROVE THE SCORE (from the model, strongest first)")
            for reason in self.top_reasons:
                lines.append(
                    f"- {reason['feature']} = {reason.get('value')} "
                    f"(contribution {reason['contribution']:+.3f})"
                )

        lines.append("")
        if self.similar_cases:
            lines.append("SIMILAR PAST CASES (already confirmed)")
            for case in self.similar_cases:
                similarity = f", similarity {case.similarity:.2f}" if case.similarity else ""
                lines.append(f"- id {case.transaction_id}: {case.summary_line()}{similarity}")
        else:
            lines.append("SIMILAR PAST CASES: none found.")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["decided_at"] = self.decided_at.isoformat()
        payload["similar_cases"] = [
            {
                "transaction_id": case.transaction_id,
                "amount": case.amount,
                "is_fraud": case.is_fraud,
                "decision": case.decision,
                "similarity": case.similarity,
            }
            for case in self.similar_cases
        ]
        return payload


def _format_number(value: float) -> str:
    """Whole numbers without a decimal tail; everything else to two places."""
    if value == int(value):
        return str(int(value))
    return f"{value:.2f}"


def build_facts(decision: Decision, similar_cases: list[PastCase] | None = None) -> CaseFacts:
    """Assemble the facts for one flagged payment from its audit record.

    The audit record is the source: whatever the analyst is shown is exactly
    what was logged at decision time, not a re-derivation that might disagree
    with it.
    """
    return CaseFacts(
        transaction_id=decision.transaction_id,
        card_token=decision.card_token,
        decided_at=decision.decided_at,
        decision=decision.decision,
        score=decision.score,
        triggered_by=decision.triggered_by,
        reason=decision.reason,
        rule_name=decision.rule_name,
        model_version=decision.model_version,
        review_threshold=decision.review_threshold,
        block_threshold=decision.block_threshold,
        amount=float(decision.features.get("amount", 0.0)),
        features=dict(decision.features),
        top_reasons=list(decision.top_reasons),
        similar_cases=list(similar_cases or []),
    )


# Matches numbers as a reader would see them: 1,234.50 / 0.87 / 12 / -3.2
_NUMBER_PATTERN = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_numbers(text: str) -> list[float]:
    """Every number in a piece of generated text."""
    found = []
    for match in _NUMBER_PATTERN.findall(text):
        try:
            found.append(float(match.replace(",", "")))
        except ValueError:  # pragma: no cover - the pattern makes this unlikely
            continue
    return found


def unsupported_numbers(text: str, facts: CaseFacts, tolerance: float = 0.01) -> list[float]:
    """Numbers in ``text`` that the facts do not support.

    Rounding is allowed - "847.20" may legitimately appear as "847" - but an
    invented figure is not. Small integers up to 10 are ignored: they are
    almost always ordinary prose ("two signals", "3 similar cases") rather
    than a claim about the payment.
    """
    supported = facts.supported_numbers()
    rounded = {round(value, 2) for value in supported} | {round(value) for value in supported}

    offenders = []
    for number in extract_numbers(text):
        if number.is_integer() and abs(number) <= 10:
            continue
        if any(abs(number - value) <= tolerance for value in supported):
            continue
        if round(number, 2) in rounded or round(number) in rounded:
            continue
        offenders.append(number)
    return offenders
