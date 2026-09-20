"""Tests for the rules engine.

The rules file is configuration an analyst edits under time pressure, so the
engine has to fail loudly on a mistake rather than silently never matching.
"""

from __future__ import annotations

import pytest
import yaml

from serving.rules import DEFAULT_RULES_PATH, RulesEngine


def engine_from(rules: list[dict], blocklist: list[str] | None = None) -> RulesEngine:
    return RulesEngine.from_config({"rules": rules, "blocklist": {"card_tokens": blocklist or []}})


def test_a_matching_rule_returns_its_action() -> None:
    engine = engine_from(
        [
            {
                "name": "big",
                "action": "review",
                "when": {"all": [{"field": "amount", "op": ">=", "value": 100}]},
            }
        ]
    )

    assert engine.evaluate({"amount": 500}).action == "review"
    assert engine.evaluate({"amount": 50}).action is None


def test_all_requires_every_condition() -> None:
    engine = engine_from(
        [
            {
                "name": "new_and_big",
                "action": "review",
                "when": {
                    "all": [
                        {"field": "card_is_new", "op": "==", "value": 1},
                        {"field": "amount", "op": ">=", "value": 1000},
                    ]
                },
            }
        ]
    )

    assert engine.evaluate({"card_is_new": 1, "amount": 2000}).action == "review"
    assert engine.evaluate({"card_is_new": 0, "amount": 2000}).action is None


def test_any_requires_only_one() -> None:
    engine = engine_from(
        [
            {
                "name": "either",
                "action": "review",
                "when": {
                    "any": [
                        {"field": "amount", "op": ">=", "value": 1000},
                        {"field": "card_txn_count_10m", "op": ">=", "value": 10},
                    ]
                },
            }
        ]
    )

    assert engine.evaluate({"amount": 10, "card_txn_count_10m": 12}).action == "review"
    assert engine.evaluate({"amount": 10, "card_txn_count_10m": 1}).action is None


def test_the_most_severe_action_wins() -> None:
    """A review rule and a block rule both firing means block."""
    engine = engine_from(
        [
            {
                "name": "soft",
                "action": "review",
                "when": {"all": [{"field": "amount", "op": ">", "value": 1}]},
            },
            {
                "name": "hard",
                "action": "block",
                "when": {"all": [{"field": "amount", "op": ">", "value": 2}]},
            },
        ]
    )

    outcome = engine.evaluate({"amount": 10})

    assert outcome.action == "block"
    assert outcome.rule_name == "hard"
    # Both are recorded: the audit log needs everything that fired.
    assert outcome.matched == ["soft", "hard"]


def test_the_blocklist_matches_on_token() -> None:
    engine = engine_from(
        [
            {
                "name": "blocked",
                "action": "block",
                "when": {"all": [{"field": "card_token", "op": "in_blocklist"}]},
            }
        ],
        blocklist=["abc123"],
    )

    assert engine.evaluate({"card_token": "abc123"}).blocks
    assert not engine.evaluate({"card_token": "other"}).blocks


def test_a_missing_field_does_not_match_rather_than_exploding() -> None:
    """A payment with no device information must still be scoreable."""
    engine = engine_from(
        [
            {
                "name": "x",
                "action": "review",
                "when": {"all": [{"field": "absent", "op": ">=", "value": 5}]},
            }
        ]
    )

    assert engine.evaluate({}).action is None


def test_null_checks_work() -> None:
    engine = engine_from(
        [
            {
                "name": "no_device",
                "action": "review",
                "when": {"all": [{"field": "device_info", "op": "is_null"}]},
            }
        ]
    )

    assert engine.evaluate({"device_info": None}).action == "review"
    assert engine.evaluate({"device_info": "Windows"}).action is None


def test_an_unknown_operator_is_rejected_at_load_time() -> None:
    """Not at match time, where it would silently never fire."""
    with pytest.raises(ValueError, match="unknown operator"):
        engine_from(
            [
                {
                    "name": "x",
                    "action": "review",
                    "when": {"all": [{"field": "a", "op": "~=", "value": 1}]},
                }
            ]
        )


def test_an_unknown_action_is_rejected() -> None:
    with pytest.raises(ValueError, match="action must be"):
        engine_from(
            [
                {
                    "name": "x",
                    "action": "shrug",
                    "when": {"all": [{"field": "a", "op": ">", "value": 1}]},
                }
            ]
        )


def test_a_rule_with_no_conditions_is_rejected() -> None:
    with pytest.raises(ValueError, match="no conditions"):
        engine_from([{"name": "x", "action": "review", "when": {"all": []}}])


def test_a_rule_without_a_when_block_is_rejected() -> None:
    with pytest.raises(ValueError, match="'all' or 'any'"):
        engine_from([{"name": "x", "action": "review"}])


# --- The shipped rules file -------------------------------------------------


def test_the_shipped_rules_file_loads() -> None:
    engine = RulesEngine.from_file()

    assert len(engine.rules) >= 4
    assert all(rule.description for rule in engine.rules), "every rule needs a why"


def test_the_shipped_rules_are_valid_yaml_with_the_expected_shape() -> None:
    config = yaml.safe_load(DEFAULT_RULES_PATH.read_text())

    assert config["version"] == 1
    assert "card_tokens" in config["blocklist"]


def test_an_ordinary_payment_matches_no_rule() -> None:
    """If everyday payments trip a rule, the review queue is useless."""
    engine = RulesEngine.from_file()

    outcome = engine.evaluate(
        {
            "amount": 42.0,
            "card_txn_count_10m": 1.0,
            "card_is_new": 0.0,
            "r_emaildomain": "gmail.com",
            "card_token": "not-on-any-list",
        }
    )

    assert outcome.action is None
    assert outcome.matched == []


def test_the_shipped_rules_catch_a_card_testing_burst() -> None:
    engine = RulesEngine.from_file()

    outcome = engine.evaluate({"amount": 12.0, "card_txn_count_10m": 15.0, "card_is_new": 0.0})

    assert outcome.action == "review"
    assert outcome.rule_name == "extreme_velocity"
