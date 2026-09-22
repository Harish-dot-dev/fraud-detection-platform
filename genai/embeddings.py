"""Turning a fraud case into a vector.

Two implementations behind one interface, for a reason worth stating:

``SentenceTransformerEmbedder`` is the real one - ``all-MiniLM-L6-v2``, 384
dimensions, small enough to run on a laptop CPU in milliseconds.

``HashingEmbedder`` is deterministic, needs no model download, and exists so
the retrieval and evaluation code can be tested in CI. It produces vectors with
the right shape and stable-but-meaningless geometry: identical text gives
identical vectors, different text gives different ones, and that is all. It is
**not** a semantic embedder and must never be used to produce a published
retrieval-quality number - the eval report records which embedder ran precisely
so that cannot happen by accident.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger("genai.embeddings")

# all-MiniLM-L6-v2's output dimension. Both implementations match it so the
# pgvector column type does not depend on which one is in use.
EMBEDDING_DIMENSIONS = 384


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into a fixed-size vector."""

    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalise(vectors: np.ndarray) -> np.ndarray:
    """Unit-length rows, so cosine similarity is a dot product."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class SentenceTransformerEmbedder:
    """The real embedder: sentence-transformers on CPU."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name, device="cpu")

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True
        )
        return np.asarray(vectors, dtype="float32")


class AzureOpenAIEmbedder:
    """The hosted embedder: an embedding deployment on Azure OpenAI.

    ``text-embedding-3-small`` returns 1536 dimensions by default, and the
    pgvector column is ``vector(384)``. Rather than migrate the schema, the
    request asks Azure for 384 directly: the text-embedding-3 models are
    trained so that a truncated prefix of the vector is still a usable
    embedding, and the API exposes that through the ``dimensions`` parameter.
    The result is that the local and hosted embedders are interchangeable in
    the same table - which is the only reason a provider switch does not mean
    reloading every case.

    Vectors from different embedders are *not* comparable even at equal length,
    so ``CaseStore`` records which embedder wrote each row and refuses to mix
    them; switching provider means reloading the store (``make load-cases``).
    """

    _RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
    # Azure rejects very large embedding batches; this is comfortably inside
    # the limit and keeps a reload of a few thousand cases to a few requests.
    _BATCH_SIZE = 64

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment: str = "text-embedding-3-small",
        api_version: str = "2024-10-21",
        dimensions: int = EMBEDDING_DIMENSIONS,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        if not endpoint or not api_key:
            raise ValueError("Azure OpenAI needs both an endpoint and an API key")

        self.name = f"azure/{deployment}"
        self.dimensions = dimensions
        self._url = (
            f"{endpoint.rstrip('/')}/openai/deployments/{deployment}"
            f"/embeddings?api-version={api_version}"
        )
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        import httpx

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = httpx.post(
                    self._url,
                    headers={"api-key": self._api_key, "Content-Type": "application/json"},
                    json={"input": texts, "dimensions": self.dimensions},
                    timeout=self._timeout,
                )
            except httpx.RequestError as error:
                last_error = error
            else:
                if response.status_code not in self._RETRY_STATUSES:
                    response.raise_for_status()
                    payload = response.json()
                    # The API does not promise that data comes back in request
                    # order - each item carries its own index. Sorting by it is
                    # what keeps case 7's vector attached to case 7 rather than
                    # to whichever case happened to be returned seventh.
                    ordered = sorted(payload["data"], key=lambda item: item["index"])
                    return np.asarray([item["embedding"] for item in ordered], dtype="float32")
                last_error = httpx.HTTPStatusError(
                    f"azure returned {response.status_code}",
                    request=response.request,
                    response=response,
                )

            if attempt < self._max_retries:
                time.sleep(2.0 * (attempt + 1))

        raise RuntimeError(f"Azure OpenAI embedding request failed after retries: {last_error}")

    def embed(self, texts: list[str]) -> np.ndarray:
        batches = [
            self._embed_batch(texts[start : start + self._BATCH_SIZE])
            for start in range(0, len(texts), self._BATCH_SIZE)
        ]
        vectors = np.vstack(batches)
        if vectors.shape[1] != self.dimensions:
            # Loud rather than silent: a vector of the wrong width would be
            # rejected by pgvector anyway, but several frames later and with a
            # message that says nothing about embeddings.
            raise ValueError(
                f"{self.name} returned {vectors.shape[1]} dimensions, expected {self.dimensions}"
            )
        # The API normalises already; doing it again is cheap and makes the
        # cosine-as-dot-product assumption hold regardless of provider.
        return _normalise(vectors)


class HashingEmbedder:
    """A deterministic stand-in with no model download.

    Hashes character n-grams into a fixed number of buckets. Same text in,
    same vector out; similar text shares some buckets. That is enough to
    exercise the vector store, the retrieval SQL and the evaluation harness,
    and nowhere near enough to call it semantic search.
    """

    name = "hashing-stub"

    def __init__(self, dimensions: int = EMBEDDING_DIMENSIONS, ngram: int = 4) -> None:
        self.dimensions = dimensions
        self.ngram = ngram

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype="float32")
        cleaned = " ".join(text.lower().split())
        if not cleaned:
            return vector

        for start in range(max(len(cleaned) - self.ngram + 1, 1)):
            token = cleaned[start : start + self.ngram].encode()
            bucket = int.from_bytes(hashlib.blake2b(token, digest_size=4).digest(), "big")
            vector[bucket % self.dimensions] += 1.0
        return vector

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.vstack([self._embed_one(text) for text in texts]))


def build_embedder(settings: Any, allow_stub: bool = False) -> Embedder:
    """Build the configured embedder, optionally falling back to the stub.

    The fallback is opt-in and local-only. Silently degrading to a meaningless
    embedder would produce a retrieval-quality number that looks real and is
    not; silently degrading away from a *paid* provider would also hide the
    fact that the run cost nothing because it never happened.
    """
    if settings.embedding_provider == "azure_openai":
        return AzureOpenAIEmbedder(
            endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key.get_secret_value(),
            deployment=settings.azure_openai_embedding_deployment,
            api_version=settings.azure_openai_api_version,
        )

    try:
        return SentenceTransformerEmbedder(settings.embedding_model)
    except Exception as error:  # noqa: BLE001
        if not allow_stub:
            raise
        logger.warning(
            "could not load %s (%s); falling back to the hashing stub. "
            "Retrieval quality measured with it is meaningless.",
            settings.embedding_model,
            error,
        )
        return HashingEmbedder()
