"""Shared vector storage schemas."""

from dataclasses import dataclass, field
from typing import Any


Metadata = dict[str, Any]
Embedding = list[float]


@dataclass(frozen=True)
class VectorRecord:
    """Embedding vector and provenance for a document chunk."""

    chunk_id: str
    document_id: str
    embedding: Embedding
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    """Vector search result with similarity score."""

    record: VectorRecord
    score: float
