"""Tests for card tokenisation."""

from __future__ import annotations

from common.pii import TOKEN_LENGTH, card_token

SALT = "a" * 64
CARD_KEY = "13926|315|gmail.com"


def test_token_is_stable_for_the_same_card_and_salt() -> None:
    """Features are per card, so the same card must always tokenise the same."""
    assert card_token(CARD_KEY, SALT) == card_token(CARD_KEY, SALT)


def test_different_cards_get_different_tokens() -> None:
    assert card_token(CARD_KEY, SALT) != card_token("13927|315|gmail.com", SALT)


def test_changing_the_salt_invalidates_every_token() -> None:
    """This is the revocation mechanism if a token set ever leaks."""
    assert card_token(CARD_KEY, SALT) != card_token(CARD_KEY, "b" * 64)


def test_token_reveals_nothing_about_the_card() -> None:
    token = card_token(CARD_KEY, SALT)

    assert len(token) == TOKEN_LENGTH
    assert "gmail" not in token
    assert "13926" not in token
    assert "315" not in token


def test_token_is_hex() -> None:
    int(card_token(CARD_KEY, SALT), 16)


def test_the_example_salt_warns_once(caplog) -> None:
    """A demo must not quietly pretend it has masked anything."""
    import common.pii

    common.pii._warned_about_default_salt = False
    with caplog.at_level("WARNING"):
        card_token(CARD_KEY, "change-me-generate-a-random-64-char-hex-string")
        card_token(CARD_KEY, "change-me-generate-a-random-64-char-hex-string")

    assert sum("PII_HASH_SALT" in record.message for record in caplog.records) == 1
