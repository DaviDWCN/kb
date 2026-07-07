"""HTTP-facing document request and response schemas."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agentic_kb.schemas.documents import DocumentStatus


@dataclass(frozen=True)
class DocumentUploadRequest:
    """Client payload for submitting a document source for ingestion."""

    source_uri: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentResponse:
    """Client-facing representation of a document record."""

    id: str
    source_uri: str
    status: DocumentStatus
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None
