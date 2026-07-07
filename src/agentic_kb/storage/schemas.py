"""Storage-specific row schemas.

These shapes describe persistence records and should not leak into domain or
API layers directly.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class DocumentRow:
    """Serialized document row shape for persistence adapters."""

    id: str
    source_uri: str
    status: str
    metadata_json: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ChunkRow:
    """Serialized chunk row shape for persistence adapters."""

    id: str
    document_id: str
    text: str
    index: int
    metadata_json: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class MetadataRow:
    """Generic metadata row attached to a stored owner record."""

    owner_id: str
    owner_type: str
    metadata_json: str
    updated_at: datetime
