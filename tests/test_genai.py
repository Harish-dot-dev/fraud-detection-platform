"""Tests for the analyst assistant.

The LLM itself is substituted - what is tested is everything around it, which
is where the engineering lives: what the model is allowed to know, whether its
output can be trusted, and what happens when it cannot.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from genai.embeddings import EMBEDDING_DIMENSIONS, HashingEmbedder
from genai.facts import CaseFacts, extract_numbers, unsupported_numbers
from genai.summarise import (
    AnalystSummary,
    ScriptedGenerator,
    check_grounding,
    parse_json_object,
    summarise_case,
)
from serving.decisions import Decision

NOW = datetime(2023, 6, 1, 12, 0, tzinfo=UTC)


@pytest.fixture
def decision() -> Decision:
    return Decision(
        transaction_id=2987123,
        card_token="a" * 32,
        decided_at=NOW,
        decision="review",
        score=0.6421,
        triggered_by="model",
        reason="model score 0.642 at or above the review threshold 0.300",
        model_version="7",
        review_threshold=0.3,
        block_threshold=0.8,
        features={
            "amount": 847.2,
            "card_txn_count_10m": 4.0,
            "card_txn_count_24h": 9.0,
            "amount_to_card_avg_ratio": 9.19,
            "card_amount_avg_lifetime": 92.15,
            "card_new_device": 1.0,
            "is_night": 1.0,
        },
        top_reasons=[
            {"feature": "amount_to_card_avg_ratio", "value": "9.19", "contribution": 0.83},
            {"feature": "card_new_device", "value": "1.0", "contribution": 0.41},
        ],
    )


@pytest.fixture
def facts(decision: Decision) -> CaseFacts:
    from genai.facts import build_facts

    return build_facts(decision)


def _summary(**overrides) -> AnalystSummary:
    payload = {
        "summary": "The payment of 847.20 is 9.19 times this card's average of 92.15, "
        "and was made on a device not previously seen on this card.",
        "key_risk_signals": ["unusually large for this card", "new device"],
        "similar_cases": [],
        "recommended_action": "investigate_further",
        "confidence": 0.7,
    }
    payload.update(overrides)
    return AnalystSummary.model_validate(payload)


# --- The facts object -------------------------------------------------------


def test_the_facts_come_from_the_audit_record(facts: CaseFacts, decision: Decision) -> None:
    """The analyst sees what was logged, not a re-derivation that might differ."""
    assert facts.transaction_id == decision.transaction_id
    assert facts.score == decision.score
    assert facts.amount == 847.2
    assert facts.model_version == "7"


def test_the_prompt_block_states_the_numbers_plainly(facts: CaseFacts) -> None:
    block = facts.to_prompt_block()

    assert "847.20" in block
    assert "payments on this card in the last 10 minutes: 4" in block
    assert "model score: 0.6421" in block
    # Labels, not column names - the model is writing for a human.
    assert "multiple of this card's average payment" in block


def test_a_number_the_facts_support_is_accepted(facts: CaseFacts) -> None:
    assert unsupported_numbers("The amount was 847.20, about 9.19x the usual 92.15", facts) == []


def test_an_invented_number_is_caught(facts: CaseFacts) -> None:
    """The whole point: a 3B model asked to divide will produce a wrong number.

    Nothing in these facts is 47.3. A summary claiming it is inventing
    arithmetic, and it would read as authoritative on an analyst's screen.
    """
    assert unsupported_numbers("This is 47.3 times the cardholder's usual amount", facts) == [47.3]


def test_rounding_is_allowed(facts: CaseFacts) -> None:
    """ "847.20" may reasonably be written as "847"."""
    assert unsupported_numbers("A payment of 847 on this card", facts) == []


def test_small_integers_in_prose_are_not_treated_as_claims(facts: CaseFacts) -> None:
    assert unsupported_numbers("There are 2 strong signals here", facts) == []


def test_numbers_are_extracted_the_way_a_reader_sees_them() -> None:
    assert extract_numbers("1,234.50 and 0.87 and -3.2") == [1234.50, 0.87, -3.2]


# --- Validation and grounding ----------------------------------------------


def test_a_well_formed_response_is_accepted(facts: CaseFacts) -> None:
    generator = ScriptedGenerator([_summary().model_dump_json()])

    result = summarise_case(facts, generator)

    assert result.usable
    assert result.attempts == 1
    assert result.summary.recommended_action == "investigate_further"


def test_json_wrapped_in_a_markdown_fence_is_still_parsed() -> None:
    parsed = parse_json_object('```json\n{"a": 1}\n```')

    assert parsed == {"a": 1}


def test_prose_instead_of_json_triggers_one_repair(facts: CaseFacts) -> None:
    generator = ScriptedGenerator(["I think this looks suspicious.", _summary().model_dump_json()])

    result = summarise_case(facts, generator)

    assert result.usable
    assert result.attempts == 2
    # The repair turn quotes the problem back to the model.
    assert "rejected" in generator.calls[-1][-1]["content"]


def test_an_ungrounded_number_is_rejected(facts: CaseFacts) -> None:
    ungrounded = _summary(summary="This payment is 47.3 times the usual amount for this card.")
    generator = ScriptedGenerator([ungrounded.model_dump_json(), _summary().model_dump_json()])

    result = summarise_case(facts, generator)

    assert result.attempts == 2
    assert result.usable


def test_a_summary_that_cannot_be_grounded_is_withheld(facts: CaseFacts) -> None:
    """Nothing is better than something wrong.

    The analyst app falls back to the facts and the SHAP reasons. An analyst
    who sees no summary knows where they stand; one who sees an invented
    figure does not.
    """
    bad = _summary(summary="This payment is 47.3 times the usual amount for this card.")
    generator = ScriptedGenerator([bad.model_dump_json(), bad.model_dump_json()])

    result = summarise_case(facts, generator)

    assert result.usable is False
    assert result.summary is None
    assert "ungrounded" in result.error


def test_a_cited_case_that_was_never_provided_is_caught(facts: CaseFacts) -> None:
    """A plausible transaction id sends an analyst hunting for nothing."""
    invented = _summary(similar_cases=["9999999"])

    numbers, cases = check_grounding(invented, facts)

    assert cases == ["9999999"]


def test_an_unknown_recommended_action_fails_validation() -> None:
    with pytest.raises(ValueError, match="recommended_action"):
        _summary(recommended_action="panic")


def test_confidence_outside_zero_to_one_fails_validation() -> None:
    with pytest.raises(ValueError):
        _summary(confidence=1.5)


def test_a_dead_llm_does_not_break_the_queue(facts: CaseFacts) -> None:
    class BrokenGenerator:
        name = "broken"

        def generate(self, messages):
            raise ConnectionError("ollama is not running")

    result = summarise_case(facts, BrokenGenerator())

    assert result.usable is False
    assert "generation failed" in result.error


def test_the_model_is_not_asked_to_calculate(facts: CaseFacts) -> None:
    """The instruction is explicit, because the failure it prevents is subtle."""
    from genai.prompts import SYSTEM_PROMPT

    assert "Never calculate" in SYSTEM_PROMPT
    assert "Use ONLY the facts provided" in SYSTEM_PROMPT


# --- Embeddings -------------------------------------------------------------


def test_the_stub_embedder_is_deterministic() -> None:
    embedder = HashingEmbedder()

    first = embedder.embed(["a card testing burst"])
    second = embedder.embed(["a card testing burst"])

    assert first.shape == (1, EMBEDDING_DIMENSIONS)
    assert (first == second).all()


def test_different_text_gets_different_vectors() -> None:
    vectors = HashingEmbedder().embed(["card testing burst", "ordinary grocery payment"])

    assert float(vectors[0] @ vectors[1]) < 0.9


def test_case_descriptions_read_as_prose() -> None:
    """The embedder was trained on sentences, not on `card_is_new=1.0`."""
    from genai.case_store import describe_case

    description = describe_case(
        amount=2400.0,
        decision="block",
        features={"card_is_new": 1.0, "card_new_device": 1.0, "amount_to_card_avg_ratio": 8.0},
    )

    assert "first payment seen on this card" in description
    assert "device not seen on this card before" in description
    assert "very large payment" in description
