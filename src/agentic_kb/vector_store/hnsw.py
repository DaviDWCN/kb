"""In-memory vector store with HNSW-shaped interface semantics.

This implementation currently performs exact cosine search over an in-memory
dict. It is useful for tests and local development while preserving the public
API expected from a future approximate HNSW backend.
"""

from math import sqrt

from agentic_kb.schemas.vectors import Embedding, Metadata, SearchResult, VectorRecord
from agentic_kb.vector_store.base import VectorStore


class HNSWVectorStore(VectorStore):
    """Small in-memory vector store implementing the VectorStore interface."""

    def __init__(self, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")
        self._dimensions = dimensions
        self._records: dict[str, VectorRecord] = {}

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def upsert(self, record: VectorRecord) -> None:
        self._validate_embedding(record.embedding)
        self._records[record.chunk_id] = record

    def delete(self, chunk_id: str) -> None:
        self._records.pop(chunk_id, None)

    def update(
        self,
        chunk_id: str,
        *,
        embedding: Embedding | None = None,
        metadata: Metadata | None = None,
    ) -> None:
        current = self._records[chunk_id]
        next_embedding = current.embedding if embedding is None else embedding
        next_metadata = current.metadata if metadata is None else metadata
        self._validate_embedding(next_embedding)
        self._records[chunk_id] = VectorRecord(
            chunk_id=current.chunk_id,
            document_id=current.document_id,
            embedding=next_embedding,
            metadata=next_metadata,
        )

    def search(
        self,
        query: Embedding,
        k: int,
        *,
        filter: Metadata | None = None,
    ) -> list[SearchResult]:
        if k <= 0:
            return []

        self._validate_embedding(query)

        results = [
            SearchResult(record=record, score=_cosine_similarity(query, record.embedding))
            for record in self._records.values()
            if _matches_filter(record.metadata, filter)
        ]
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:k]

    def get(self, chunk_id: str) -> VectorRecord | None:
        return self._records.get(chunk_id)

    def __len__(self) -> int:
        return len(self._records)

    def _validate_embedding(self, embedding: Embedding) -> None:
        if len(embedding) != self._dimensions:
            raise ValueError(
                f"embedding has {len(embedding)} dimensions, expected {self._dimensions}"
            )


def _matches_filter(metadata: Metadata, filter: Metadata | None) -> bool:
    if filter is None:
        return True
    return all(metadata.get(key) == value for key, value in filter.items())


def _cosine_similarity(left: Embedding, right: Embedding) -> float:
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    return dot_product / (left_norm * right_norm)
