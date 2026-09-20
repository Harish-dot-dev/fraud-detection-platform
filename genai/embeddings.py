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
from typing import Protocol, runtime_checkable

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


def build_embedder(model_name: str, allow_stub: bool = False) -> Embedder:
    """Load the real embedder, optionally falling back to the stub.

    The fallback is opt-in. Silently degrading to a meaningless embedder would
    produce a retrieval-quality number that looks real and is not.
    """
    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception as error:  # noqa: BLE001
        if not allow_stub:
            raise
        logger.warning(
            "could not load %s (%s); falling back to the hashing stub. "
            "Retrieval quality measured with it is meaningless.",
            model_name,
            error,
        )
        return HashingEmbedder()
