"""Embedding subsystem schemas for internal jobs and batches."""

from dataclasses import dataclass, field
from datetime import datetime

from agentic_kb.schemas.vectors import Embedding


@dataclass(frozen=True)
class EmbeddingBatch:
    """Batch of texts submitted together to an embedding model."""

    id: str
    texts: list[str]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingResult:
    """Embedding output for one text item in a batch."""

    text_index: int
    embedding: Embedding
    model_name: str


@dataclass(frozen=True)
class EmbeddingJob:
    """Embedding job metadata used by workers or refresh tasks."""

    id: str
    batch: EmbeddingBatch
    model_name: str
    submitted_at: datetime | None = None
