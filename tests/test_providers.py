"""Tests for the pluggable LLM and embedding providers.

No network and no Azure subscription: ``httpx.post`` is substituted so the
tests can assert on the exact request that would have gone out. That is the
part worth pinning, because the failures this code is written against - a
wrong deployment name, a reordered embedding batch, a mismatched vector store -
all look like success from the calling side.
"""

from __future__ import annotations

from typing import Any

import httpx
import numpy as np
import pytest

from common.config import Settings
from genai.embeddings import (
    EMBEDDING_DIMENSIONS,
    AzureOpenAIEmbedder,
    HashingEmbedder,
    build_embedder,
)
from genai.prompts import SYSTEM_PROMPT
from genai.summarise import AzureOpenAIGenerator, OllamaGenerator, build_generator

ENDPOINT = "https://fraud-demo.openai.azure.com"
KEY = "test-key-never-real"


def _azure_settings(**overrides: Any) -> Settings:
    """Settings with Azure configured, ignoring any .env on the machine."""
    base = {
        "_env_file": None,
        "llm_provider": "azure_openai",
        "embedding_provider": "azure_openai",
        "azure_openai_endpoint": ENDPOINT,
        "azure_openai_api_key": KEY,
    }
    return Settings(**{**base, **overrides})


class _Recorder:
    """Stands in for httpx.post and records what it was asked to send."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url: str, **kwargs: Any) -> httpx.Response:
        self.calls.append({"url": url, **kwargs})
        response = self._responses.pop(0)
        # httpx responses need a request attached before raise_for_status works.
        response._request = httpx.Request("POST", url)
        return response


def _chat_response(content: str, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status, json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
    )


def _embedding_response(vectors: list[list[float]], indexes: list[int] | None = None):
    indexes = list(range(len(vectors))) if indexes is None else indexes
    return httpx.Response(
        200,
        json={
            "data": [
                {"index": index, "embedding": vector}
                for index, vector in zip(indexes, vectors, strict=True)
            ]
        },
    )


# --- configuration ---------------------------------------------------------


def test_default_configuration_is_local_and_free() -> None:
    """A fresh clone must not need an Azure subscription to run."""
    settings = Settings(_env_file=None)

    assert settings.llm_provider == "ollama"
    assert settings.embedding_provider == "sentence_transformers"
    assert not settings.uses_paid_provider


def test_azure_without_credentials_fails_at_startup() -> None:
    """Not at the first analyst request, thirty seconds into a demo."""
    with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
        Settings(_env_file=None, llm_provider="azure_openai")

    with pytest.raises(ValueError, match="AZURE_OPENAI_API_KEY"):
        Settings(_env_file=None, llm_provider="azure_openai", azure_openai_endpoint=ENDPOINT)


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        Settings(_env_file=None, llm_provider="gpt5-please")


def test_api_key_does_not_leak_through_repr() -> None:
    """A settings object ends up in tracebacks and Streamlit error pages."""
    settings = _azure_settings()

    assert KEY not in repr(settings)
    assert KEY not in str(settings.model_dump())
    assert settings.azure_openai_api_key.get_secret_value() == KEY


def test_paid_provider_is_flagged() -> None:
    assert _azure_settings().uses_paid_provider
    assert _azure_settings(llm_provider="ollama").uses_paid_provider  # embedder still Azure


# --- factories -------------------------------------------------------------


def test_build_generator_dispatches_on_provider() -> None:
    assert isinstance(build_generator(Settings(_env_file=None)), OllamaGenerator)
    assert isinstance(build_generator(_azure_settings()), AzureOpenAIGenerator)


def test_build_embedder_dispatches_on_provider() -> None:
    embedder = build_embedder(_azure_settings())

    assert isinstance(embedder, AzureOpenAIEmbedder)


def test_azure_embedder_never_degrades_to_the_stub() -> None:
    """allow_stub covers a missing local model download, not a paid API.

    Falling back here would report a retrieval-quality number for a run that
    silently never reached Azure at all.
    """
    embedder = build_embedder(_azure_settings(), allow_stub=True)

    assert not isinstance(embedder, HashingEmbedder)


# --- the chat generator ----------------------------------------------------


def test_generator_calls_the_deployment_url(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder([_chat_response('{"ok": true}')])
    monkeypatch.setattr(httpx, "post", recorder)

    generator = AzureOpenAIGenerator(ENDPOINT, KEY, deployment="gpt-4o-mini-prod")
    generator.generate([{"role": "user", "content": "hello"}])

    call = recorder.calls[0]
    assert call["url"] == (
        f"{ENDPOINT}/openai/deployments/gpt-4o-mini-prod" "/chat/completions?api-version=2024-10-21"
    )
    # Azure uses its own header rather than an Authorization bearer token.
    assert call["headers"]["api-key"] == KEY
    assert call["json"]["response_format"] == {"type": "json_object"}


def test_generator_is_named_for_the_deployment() -> None:
    """Reports record this string, so it has to identify the actual run."""
    assert AzureOpenAIGenerator(ENDPOINT, KEY, deployment="gpt-4o-mini").name == (
        "azure/gpt-4o-mini"
    )


def test_generator_rejects_missing_credentials() -> None:
    with pytest.raises(ValueError):
        AzureOpenAIGenerator("", KEY)
    with pytest.raises(ValueError):
        AzureOpenAIGenerator(ENDPOINT, "")


def test_system_prompt_mentions_json() -> None:
    """response_format=json_object requires it, and Azure 400s without it.

    Pinned here so rewording the prompt cannot quietly break the hosted path
    while the local path keeps working.
    """
    assert "JSON" in SYSTEM_PROMPT


def test_content_filter_raises_rather_than_returning_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fraud case text is exactly what trips a content filter."""
    blocked = httpx.Response(
        200, json={"choices": [{"message": {"content": None}, "finish_reason": "content_filter"}]}
    )
    monkeypatch.setattr(httpx, "post", _Recorder([blocked]))

    with pytest.raises(RuntimeError, match="content filter"):
        AzureOpenAIGenerator(ENDPOINT, KEY).generate([{"role": "user", "content": "x"}])


def test_throttling_is_retried_and_honours_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """429 is normal traffic on a small deployment, not an outage."""
    throttled = httpx.Response(429, headers={"retry-after": "0"}, json={})
    recorder = _Recorder([throttled, _chat_response('{"ok": true}')])
    monkeypatch.setattr(httpx, "post", recorder)

    slept: list[float] = []
    monkeypatch.setattr("genai.summarise.time.sleep", slept.append)

    content = AzureOpenAIGenerator(ENDPOINT, KEY).generate([{"role": "user", "content": "x"}])

    assert content == '{"ok": true}'
    assert len(recorder.calls) == 2
    assert slept == [0.0], "Retry-After from the response should win over the default backoff"


def test_persistent_failure_raises_after_the_retry_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [httpx.Response(503, json={}) for _ in range(3)]
    monkeypatch.setattr(httpx, "post", _Recorder(responses))
    monkeypatch.setattr("genai.summarise.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="failed after retries"):
        AzureOpenAIGenerator(ENDPOINT, KEY, max_retries=2).generate([{"role": "user", "x": "y"}])


# --- the embedder ----------------------------------------------------------


def test_embedder_asks_azure_for_the_pgvector_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """384, not the model's native 1536, so the existing column still fits."""
    recorder = _Recorder([_embedding_response([[0.1] * EMBEDDING_DIMENSIONS])])
    monkeypatch.setattr(httpx, "post", recorder)

    AzureOpenAIEmbedder(ENDPOINT, KEY).embed(["a case"])

    assert recorder.calls[0]["json"]["dimensions"] == EMBEDDING_DIMENSIONS


def test_embeddings_are_reordered_by_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """The API does not promise response order; each item carries its index.

    Without the sort this passes silently and every case in the store ends up
    holding some other case's vector.
    """
    first = [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)
    second = [0.0, 1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 2)
    # Returned back to front, as Azure is entitled to do.
    response = _embedding_response([second, first], indexes=[1, 0])
    monkeypatch.setattr(httpx, "post", _Recorder([response]))

    vectors = AzureOpenAIEmbedder(ENDPOINT, KEY).embed(["first", "second"])

    assert vectors[0][0] == pytest.approx(1.0)
    assert vectors[1][1] == pytest.approx(1.0)


def test_wrong_dimension_count_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deployment of an older model ignores `dimensions` and returns 1536."""
    monkeypatch.setattr(httpx, "post", _Recorder([_embedding_response([[0.1] * 1536])]))

    with pytest.raises(ValueError, match="dimensions"):
        AzureOpenAIEmbedder(ENDPOINT, KEY).embed(["a case"])


def test_embeddings_come_back_unit_length(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cosine similarity is a dot product downstream, for either provider."""
    monkeypatch.setattr(
        httpx,
        "post",
        _Recorder([_embedding_response([[3.0] + [4.0] * (EMBEDDING_DIMENSIONS - 1)])]),
    )

    vectors = AzureOpenAIEmbedder(ENDPOINT, KEY).embed(["a case"])

    assert np.linalg.norm(vectors[0]) == pytest.approx(1.0, abs=1e-6)


def test_large_input_is_batched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Azure rejects oversized embedding requests; a reload must not hit that."""
    texts = [f"case {index}" for index in range(150)]
    responses = [
        _embedding_response([[0.1] * EMBEDDING_DIMENSIONS] * size) for size in (64, 64, 22)
    ]
    recorder = _Recorder(responses)
    monkeypatch.setattr(httpx, "post", recorder)

    vectors = AzureOpenAIEmbedder(ENDPOINT, KEY).embed(texts)

    assert len(recorder.calls) == 3
    assert vectors.shape == (150, EMBEDDING_DIMENSIONS)


# --- the vector store guard ------------------------------------------------


class _FakeCursor:
    """Answers the one query check_embedder makes."""

    def __init__(self, stored: list[str]) -> None:
        self._stored = stored

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def execute(self, *_: Any, **__: Any) -> None:
        return None

    def fetchall(self) -> list[tuple[str]]:
        return [(name,) for name in self._stored]


class _FakeConnection:
    def __init__(self, stored: list[str]) -> None:
        self._stored = stored

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._stored)


def _store(stored: list[str], embedder_name: str):
    from genai.case_store import CaseStore

    embedder = HashingEmbedder()
    embedder.name = embedder_name  # type: ignore[misc]
    return CaseStore(_FakeConnection(stored), embedder)


def test_retrieval_refuses_a_store_written_by_another_embedder() -> None:
    """The failure mode a provider switch introduces.

    Both embedders emit 384 unit-length floats, so pgvector accepts the query
    and cosine similarity returns a plausible number about nothing at all.
    Nothing downstream can tell - so the store refuses instead.
    """
    store = _store(["azure/text-embedding-3-small"], "sentence-transformers/all-MiniLM-L6-v2")

    with pytest.raises(RuntimeError, match="not comparable"):
        store.check_embedder()


def test_matching_embedder_passes_and_is_checked_once() -> None:
    store = _store(["azure/text-embedding-3-small"], "azure/text-embedding-3-small")

    store.check_embedder()
    store.check_embedder()  # cached; must not raise or re-query


def test_rows_from_before_the_column_existed_do_not_block_retrieval() -> None:
    """'unknown' means a store loaded by an older build, not a mismatch."""
    _store(["unknown"], "azure/text-embedding-3-small").check_embedder()


def test_empty_store_passes() -> None:
    _store([], "azure/text-embedding-3-small").check_embedder()
