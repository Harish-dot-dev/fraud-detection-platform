"""Pseudonymisation of the card identity.

The card identity proxy (``card1|addr1|P_emaildomain``) identifies a person's
card. It has no business reaching a model, a dashboard, an LLM prompt or a
Redis dump, so everything downstream of ingestion sees a token instead.

The token is an HMAC-SHA256 of the card key under a secret salt, truncated to
32 hex characters.

Why HMAC rather than a plain hash: the card key comes from a small space
(``card1`` has a few thousand values, ``addr1`` a few hundred, and there are
maybe sixty email domains). A plain SHA-256 of that would be trivially
reversible by generating every combination. Keying the hash with a secret the
attacker does not have removes that attack.

This is pseudonymisation, not anonymisation. The same card always produces the
same token - which is the point, since the features are per card - so tokens
are still personal data under GDPR and still need protecting. What they do
provide is that a leaked model, feature store or dashboard extract contains no
card numbers, addresses or email domains.
"""

from __future__ import annotations

import hashlib
import hmac
import logging

logger = logging.getLogger(__name__)

TOKEN_LENGTH = 32
_DEFAULT_SALT_PREFIX = "change-me"
_warned_about_default_salt = False


def card_token(card_key: str, salt: str) -> str:
    """Tokenise a card identity proxy.

    Args:
        card_key: the raw proxy, e.g. ``"13926|315|gmail.com"``.
        salt: the secret from ``PII_HASH_SALT``.

    Returns:
        32 hex characters, stable for a given (card_key, salt) pair.

    Changing the salt invalidates every previously issued token, which means
    the online feature store and the Gold tables have to be rebuilt. That is a
    deliberate property: it is how you revoke tokens if one ever leaks.
    """
    global _warned_about_default_salt

    if salt.startswith(_DEFAULT_SALT_PREFIX) and not _warned_about_default_salt:
        # Loud once, not once per row.
        logger.warning(
            "PII_HASH_SALT is still the example value: tokens are reproducible "
            "by anyone with this repository. Generate one with "
            '`python -c "import secrets; print(secrets.token_hex(32))"`.'
        )
        _warned_about_default_salt = True

    digest = hmac.new(salt.encode("utf-8"), card_key.encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()[:TOKEN_LENGTH]
