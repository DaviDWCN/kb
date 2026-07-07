"""Shared chunk domain schemas."""

from dataclasses import dataclass, field
from typing import Any

from agentic_kb.schemas.documents import DocumentId


ChunkId = str


@dataclass(frozen=True)
class ChunkMetadata:
    """Metadata carried alongside chunk text for citation and filtering."""

    heading_path: tuple[str, ...] = ()
    page_number: int | None = None
    token_count: int | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """Canonical text chunk consumed by embeddings and retrieval."""

    id: ChunkId
    document_id: DocumentId
    text: str
    index: int
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)
