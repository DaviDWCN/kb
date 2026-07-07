"""Shared document domain schemas."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from agentic_kb.schemas.metadata import DocumentMetadata


DocumentId = str


class DocumentStatus(StrEnum):
    """Lifecycle states for a document in the knowledge base."""

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    DELETED = "deleted"


@dataclass(frozen=True)
class Document:
    """Canonical document record independent of API or storage details."""

    id: DocumentId
    source_uri: str
    status: DocumentStatus = DocumentStatus.PENDING
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    error: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
