"""Shared search and citation domain schemas."""

from dataclasses import dataclass, field
from typing import Any

from agentic_kb.schemas.chunks import Chunk, ChunkId
from agentic_kb.schemas.documents import DocumentId


SearchFilters = dict[str, Any]


@dataclass(frozen=True)
class SearchQuery:
    """Internal search query contract used by retrieval components."""

    text: str
    limit: int = 10
    filters: SearchFilters = field(default_factory=dict)


@dataclass(frozen=True)
class Citation:
    """Reference back to the source document/chunk for a retrieved result."""

    document_id: DocumentId
    chunk_id: ChunkId
    score: float
    span: tuple[int, int] | None = None


@dataclass(frozen=True)
class SearchHit:
    """Retrieved chunk plus score and optional citation/highlights."""

    chunk: Chunk
    score: float
    citation: Citation | None = None
    highlights: tuple[str, ...] = ()
