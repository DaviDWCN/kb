"""HTTP-facing ingestion job schemas."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class IngestionJobResponse:
    """Client-facing status for an asynchronous ingestion job."""

    id: str
    document_id: str | None
    status: str
    submitted_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
