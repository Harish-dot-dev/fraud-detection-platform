"""Prompts for the analyst assistant.

Written to a specific brief: the reader is a fraud analyst with a queue to get
through, who needs to decide in under a minute and be able to justify it
afterwards. Not a chatbot, not a summary of the summary.

Three constraints do most of the work:

1. **Use only the facts given.** No outside knowledge about fraud patterns, no
   inference about the cardholder, no arithmetic.
2. **Say when something is not supported.** An analyst can work with "the facts
   do not say"; they cannot work with a confident invention.
3. **Return JSON matching the schema.** Free text would have to be parsed, and
   a parse that half-works is worse than one that fails.
"""

from __future__ import annotations

from genai.facts import CaseFacts

SYSTEM_PROMPT = """You are a fraud analysis assistant for payment reviewers.

Your job is to turn the facts you are given into a short, plain summary that \
helps an analyst decide quickly.

Rules you must follow:
1. Use ONLY the facts provided. Never invent a number, a date, a merchant, a \
location or a cardholder detail.
2. Never calculate anything. Every number you need has been provided; quote it \
as given.
3. If the facts do not support something, say so plainly rather than guessing.
4. Do not claim the payment is fraud. A human decides that. Describe what the \
signals show.
5. Reply with a single JSON object and nothing else - no markdown fence, no \
commentary before or after.

The JSON object must have exactly these fields:
{
  "summary": "2-3 sentences an analyst can read in ten seconds",
  "key_risk_signals": ["short phrases, strongest first, at most 5"],
  "similar_cases": ["transaction ids from the SIMILAR PAST CASES section, as strings"],
  "recommended_action": one of "approve", "reject", "investigate_further",
  "confidence": a number between 0 and 1 for how well the facts support your reading
}"""

USER_PROMPT_TEMPLATE = """{facts}

Write the JSON object now. Remember: only the facts above, no arithmetic, and \
say so if something is unsupported."""


def build_messages(facts: CaseFacts) -> list[dict[str, str]]:
    """The chat messages for one flagged payment."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(facts=facts.to_prompt_block())},
    ]


def build_repair_message(error: str) -> dict[str, str]:
    """A single corrective turn after an invalid response.

    One retry, with the specific problem quoted back. More than one and a
    small model tends to drift further rather than converge, and the analyst
    is waiting.
    """
    return {
        "role": "user",
        "content": (
            f"That response was rejected: {error}\n\n"
            "Reply again with a single valid JSON object that follows the rules exactly. "
            "Use only the facts already given, and do not calculate anything."
        ),
    }
