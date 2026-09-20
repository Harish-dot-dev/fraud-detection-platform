"""A small, configuration-driven rules engine.

Rules live in YAML (``rules/rules.yaml``) so an analyst can add one without a
deployment, and they run before the model because some decisions are policy
rather than prediction - a confirmed-compromised card, a regulatory limit, a
brand-new attack pattern that cannot wait a week for a retrain.

The condition language is deliberately tiny: a field, an operator and a value,
combined with ``all`` or ``any``. There is **no ``eval``** anywhere in it. A
rules file is configuration, and configuration that can execute arbitrary
Python is a remote code execution bug waiting for someone to edit the wrong
file.

Unknown operators and unknown fields fail loudly at load time rather than
silently never matching, which is the failure mode that lets a rule sit there
for months doing nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from common.config import REPO_ROOT

DEFAULT_RULES_PATH = REPO_ROOT / "rules" / "rules.yaml"

ACTION_BLOCK = "block"
ACTION_REVIEW = "review"
VALID_ACTIONS = {ACTION_BLOCK, ACTION_REVIEW}

# Ordered by severity so decisions can be compared.
ACTION_SEVERITY = {None: 0, ACTION_REVIEW: 1, ACTION_BLOCK: 2}


def _op_in_blocklist(value: Any, _expected: Any, blocklist: set[str]) -> bool:
    return value is not None and str(value) in blocklist


OPERATORS = {
    ">": lambda value, expected, _b: value is not None and value > expected,
    ">=": lambda value, expected, _b: value is not None and value >= expected,
    "<": lambda value, expected, _b: value is not None and value < expected,
    "<=": lambda value, expected, _b: value is not None and value <= expected,
    "==": lambda value, expected, _b: value == expected,
    "!=": lambda value, expected, _b: value != expected,
    "in": lambda value, expected, _b: value in expected,
    "not_in": lambda value, expected, _b: value not in expected,
    "is_null": lambda value, _e, _b: value is None,
    "not_null": lambda value, _e, _b: value is not None,
    "in_blocklist": _op_in_blocklist,
}


@dataclass(frozen=True)
class Condition:
    """One field/operator/value test."""

    field: str
    op: str
    value: Any = None

    def matches(self, context: dict[str, Any], blocklist: set[str]) -> bool:
        return OPERATORS[self.op](context.get(self.field), self.value, blocklist)


@dataclass(frozen=True)
class Rule:
    """A named rule: conditions plus the action to take when they hold."""

    name: str
    action: str
    conditions: tuple[Condition, ...]
    mode: str = "all"
    description: str = ""

    def matches(self, context: dict[str, Any], blocklist: set[str]) -> bool:
        results = (condition.matches(context, blocklist) for condition in self.conditions)
        return all(results) if self.mode == "all" else any(results)


@dataclass
class RuleOutcome:
    """What the rules made of one payment."""

    action: str | None = None
    rule_name: str | None = None
    description: str = ""
    # Every rule that matched, not just the winner - useful in the audit log.
    matched: list[str] = field(default_factory=list)

    @property
    def blocks(self) -> bool:
        return self.action == ACTION_BLOCK


class RulesEngine:
    """Evaluates the rules file against a payment."""

    def __init__(self, rules: list[Rule], blocklist: set[str] | None = None) -> None:
        self.rules = rules
        self.blocklist = blocklist or set()

    @classmethod
    def from_file(cls, path: str | Path = DEFAULT_RULES_PATH) -> RulesEngine:
        return cls.from_config(yaml.safe_load(Path(path).read_text()) or {})

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RulesEngine:
        """Build an engine from parsed YAML, validating as it goes."""
        rules = []
        for index, raw in enumerate(config.get("rules") or []):
            name = raw.get("name") or f"rule_{index}"
            action = raw.get("action")
            if action not in VALID_ACTIONS:
                raise ValueError(f"rule {name!r}: action must be one of {sorted(VALID_ACTIONS)}")

            when = raw.get("when") or {}
            mode = "all" if "all" in when else "any"
            if mode not in when:
                raise ValueError(f"rule {name!r}: 'when' needs an 'all' or 'any' block")

            conditions = []
            for raw_condition in when[mode]:
                op = raw_condition.get("op")
                if op not in OPERATORS:
                    raise ValueError(
                        f"rule {name!r}: unknown operator {op!r}; "
                        f"expected one of {sorted(OPERATORS)}"
                    )
                conditions.append(
                    Condition(field=raw_condition["field"], op=op, value=raw_condition.get("value"))
                )

            if not conditions:
                raise ValueError(f"rule {name!r}: no conditions")

            rules.append(
                Rule(
                    name=name,
                    action=action,
                    conditions=tuple(conditions),
                    mode=mode,
                    description=(raw.get("description") or "").strip(),
                )
            )

        blocklist = set((config.get("blocklist") or {}).get("card_tokens") or [])
        return cls(rules, blocklist)

    def evaluate(self, context: dict[str, Any]) -> RuleOutcome:
        """Apply every rule; the most severe action wins.

        All rules are evaluated even once one has matched - the audit log
        records everything that fired, which is what makes it possible to
        answer "why was this blocked?" months later.
        """
        outcome = RuleOutcome()

        for rule in self.rules:
            if not rule.matches(context, self.blocklist):
                continue

            outcome.matched.append(rule.name)
            if ACTION_SEVERITY[rule.action] > ACTION_SEVERITY[outcome.action]:
                outcome.action = rule.action
                outcome.rule_name = rule.name
                outcome.description = rule.description

        return outcome
