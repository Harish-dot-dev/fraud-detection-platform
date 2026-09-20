"""Generating a grounded case summary with a local LLM.

The pipeline is deliberately defensive, because a 3B model running on a laptop
is a useful writer and an unreliable arithmetician:

    facts -> retrieve similar cases -> prompt -> generate -> parse JSON
          -> validate against a Pydantic schema
          -> check every number against the facts
          -> one repair attempt if any of that failed
          -> return the summary, or nothing at all

**Nothing is better than something wrong.** If the output cannot be validated
and grounded, the analyst app shows the facts and the SHAP reasons without a
narrative. An analyst who sees no summary knows where they stand; one who sees
a confident invented figure does not.

This runs only for review and block decisions - never for the ~97% of payments
that are allowed. Nobody reads a summary of a payment that went through, and
generation costs seconds rather than milliseconds.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError, field_validator

from genai.facts import CaseFacts, unsupported_numbers
from genai.prompts import build_messages, build_repair_message

logger = logging.getLogger("genai.summarise")

RECOMMENDED_ACTIONS = ("approve", "reject", "investigate_further")


class AnalystSummary(BaseModel):
    """The structured output an analyst sees."""

    summary: str = Field(min_length=20, max_length=1200)
    key_risk_signals: list[str] = Field(default_factory=list, max_length=5)
    similar_cases: list[str] = Field(default_factory=list, max_length=10)
    recommended_action: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("recommended_action")
    @classmethod
    def _known_action(cls, value: str) -> str:
        normalised = value.strip().lower().replace(" ", "_")
        if normalised not in RECOMMENDED_ACTIONS:
            raise ValueError(f"recommended_action must be one of {RECOMMENDED_ACTIONS}")
        return normalised

    @field_validator("key_risk_signals", "similar_cases")
    @classmethod
    def _no_empty_strings(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item and item.strip()]


@dataclass
class SummaryResult:
    """What came back, and whether it can be trusted."""

    summary: AnalystSummary | None
    valid: bool
    grounded: bool
    attempts: int
    latency_ms: float
    model: str = ""
    error: str = ""
    # Numbers in the text that the facts did not support.
    unsupported: list[float] = field(default_factory=list)
    # Retrieved case ids the model claimed but was never given.
    invented_case_ids: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """Only a valid *and* grounded summary is shown to an analyst."""
        return self.valid and self.grounded and self.summary is not None


class SummaryGenerator(Protocol):
    """Anything that can answer a chat prompt."""

    name: str

    def generate(self, messages: list[dict[str, str]]) -> str: ...


class OllamaGenerator:
    """The real generator: a local model served by Ollama."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
        timeout: float = 120.0,
        temperature: float = 0.1,
    ) -> None:
        self.name = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        # Low, not zero: the task is near-extractive, and a little variation
        # helps the repair attempt produce something different from the first.
        self._temperature = temperature

    def generate(self, messages: list[dict[str, str]]) -> str:
        import httpx

        response = httpx.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self.name,
                "messages": messages,
                "stream": False,
                # Ollama can constrain decoding to JSON, which removes a whole
                # class of parse failures before they happen.
                "format": "json",
                "options": {"temperature": self._temperature},
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


class ScriptedGenerator:
    """Returns prepared responses. For tests and for the eval harness offline."""

    name = "scripted"

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if not self._responses:
            raise RuntimeError("ScriptedGenerator ran out of responses")
        return self._responses.pop(0)


# A model may still wrap JSON in a fence despite being told not to.
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(text)
        if not match:
            raise ValueError("no JSON object in the response") from None
        return json.loads(match.group(0))


def check_grounding(summary: AnalystSummary, facts: CaseFacts) -> tuple[list[float], list[str]]:
    """Find anything in the summary the facts do not support.

    Two kinds of invention are checked: numbers that appear nowhere in the
    facts, and similar-case ids the model was never shown. The second is the
    subtler one - a plausible-looking transaction id that does not exist sends
    an analyst hunting for a case that was never there.
    """
    text = " ".join([summary.summary, *summary.key_risk_signals])
    offending_numbers = unsupported_numbers(text, facts)

    offered = {str(case_id) for case_id in facts.similar_case_ids}
    invented = [case_id for case_id in summary.similar_cases if str(case_id).strip() not in offered]
    return offending_numbers, invented


def summarise_case(
    facts: CaseFacts, generator: SummaryGenerator, allow_repair: bool = True
) -> SummaryResult:
    """Generate, validate and ground a summary for one flagged payment."""
    messages = build_messages(facts)
    started = time.perf_counter()
    attempts = 0
    last_error = ""

    while attempts < (2 if allow_repair else 1):
        attempts += 1
        try:
            raw = generator.generate(messages)
        except Exception as error:  # noqa: BLE001 - a dead LLM must not break the queue
            last_error = f"generation failed: {error}"
            logger.warning(last_error)
            break

        try:
            summary = AnalystSummary.model_validate(parse_json_object(raw))
        except (ValueError, ValidationError) as error:
            last_error = f"invalid response: {str(error)[:200]}"
            logger.info("attempt %s rejected (%s)", attempts, last_error)
            messages = [
                *messages,
                {"role": "assistant", "content": raw},
                build_repair_message(last_error),
            ]
            continue

        unsupported, invented = check_grounding(summary, facts)
        if unsupported or invented:
            last_error = (
                f"ungrounded: numbers {unsupported} not in the facts"
                if unsupported
                else f"ungrounded: cited cases {invented} were never provided"
            )
            logger.info("attempt %s rejected (%s)", attempts, last_error)
            messages = [
                *messages,
                {"role": "assistant", "content": raw},
                build_repair_message(last_error),
            ]
            continue

        return SummaryResult(
            summary=summary,
            valid=True,
            grounded=True,
            attempts=attempts,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            model=generator.name,
        )

    # Everything failed. The analyst app falls back to facts and SHAP reasons.
    return SummaryResult(
        summary=None,
        valid=False,
        grounded=False,
        attempts=attempts,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        model=generator.name,
        error=last_error,
    )
