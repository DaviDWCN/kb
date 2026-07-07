"""High-level embedding orchestration."""

from agentic_kb.embeddings.models import EmbeddingModel
from agentic_kb.embeddings.schemas import EmbeddingBatch, EmbeddingResult
from agentic_kb.schemas.vectors import Embedding


class EmbeddingService:
    """Coordinates validation, batching, and result shaping around a model."""

    def __init__(self, model: EmbeddingModel, max_batch_size: int = 128) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than zero")

        self._model = model
        self._max_batch_size = max_batch_size

    def embed_batch(self, batch: EmbeddingBatch) -> list[EmbeddingResult]:
        embeddings = self.embed_texts(batch.texts)
        return [
            EmbeddingResult(
                text_index=index,
                embedding=embedding,
                model_name=self._model.model_name,
            )
            for index, embedding in enumerate(embeddings)
        ]

    def embed_texts(self, texts: list[str]) -> list[Embedding]:
        self._validate_texts(texts)

        embeddings: list[Embedding] = []
        for text_batch in _iter_batches(texts, self._max_batch_size):
            batch_embeddings = self._model.embed_texts(text_batch)
            self._validate_provider_response(text_batch, batch_embeddings)
            embeddings.extend(batch_embeddings)

        return embeddings

    def _validate_texts(self, texts: list[str]) -> None:
        for index, text in enumerate(texts):
            if not text.strip():
                raise ValueError(f"text at index {index} is empty")

    def _validate_provider_response(
        self,
        texts: list[str],
        embeddings: list[Embedding],
    ) -> None:
        if len(embeddings) != len(texts):
            raise ValueError(
                f"{self._model.model_name} returned {len(embeddings)} embeddings "
                f"for {len(texts)} texts"
            )

        for index, embedding in enumerate(embeddings):
            if len(embedding) != self._model.dimensions:
                raise ValueError(
                    f"{self._model.model_name} returned embedding {index} with "
                    f"{len(embedding)} dimensions, expected {self._model.dimensions}"
                )


def _iter_batches(items: list[str], batch_size: int) -> list[list[str]]:
    """Split texts into provider-sized batches while preserving order."""
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]
