"""Embedding model interfaces and local implementations."""

import hashlib
import math
from abc import ABC, abstractmethod

from agentic_kb.schemas.vectors import Embedding


class EmbeddingModel(ABC):
    """Provider-neutral contract for anything that can embed text."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Stable model identifier stored with embedding results."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Number of dimensions returned by the model."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[Embedding]:
        """Embed multiple texts in one provider call."""

    def embed_text(self, text: str) -> Embedding:
        """Convenience wrapper for single-text callers."""
        return self.embed_texts([text])[0]


class HashEmbeddingModel(EmbeddingModel):
    """Hash-based local model intended for repeatable tests and development.

    This model is deliberately not semantic: similar texts are not guaranteed to
    produce similar vectors. Production providers should implement EmbeddingModel
    with a real embedding backend.
    """

    def __init__(self, dimensions: int, model_name: str = "hash-embedding") -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")

        self._dimensions = dimensions
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_texts(self, texts: list[str]) -> list[Embedding]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> Embedding:
        """Map text to a stable unit vector without external model dependencies."""
        values: list[float] = []
        seed = text.encode("utf-8")

        # SHA-256 emits 32 bytes, so extend it with a counter for larger dimensions.
        counter = 0
        while len(values) < self._dimensions:
            digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            values.extend((byte / 127.5) - 1.0 for byte in digest)
            counter += 1

        vector = values[: self._dimensions]
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            return [0.0] * self._dimensions

        return [value / magnitude for value in vector]
