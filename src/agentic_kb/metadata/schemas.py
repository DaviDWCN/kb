"""Metadata extraction subsystem schemas."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MetadataExtractionResult:
    """Normalized result from a metadata extractor."""

    document_id: str
    metadata: dict[str, Any]
    confidence: float | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class LLMMetadataResponse:
    """Raw and parsed metadata returned by an LLM-based extractor."""

    metadata: dict[str, Any]
    raw_response: str
    model_name: str


@dataclass(frozen=True)
class MetadataExtractorConfig:
    """Configuration for a metadata extractor implementation."""

    extractor_name: str
    options: dict[str, Any] = field(default_factory=dict)
