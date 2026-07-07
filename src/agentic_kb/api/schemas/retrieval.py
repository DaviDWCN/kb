"""HTTP-facing retrieval request and response schemas."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SearchRequest:
    """Client payload for a search request."""

    query: str
    limit: int = 10
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResultResponse:
    """Client-facing search hit with text, score, and optional citation data."""

    document_id: str
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    citation: dict[str, Any] | None = None


@dataclass(frozen=True)
class SearchResponse:
    """Client-facing search response for a single query."""

    query: str
    results: list[SearchResultResponse] = field(default_factory=list)
