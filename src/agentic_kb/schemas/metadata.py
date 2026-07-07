"""Shared metadata schema for parsed or ingested documents."""

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass(frozen=True)
class DocumentMetadata:
    """Common descriptive metadata that can be attached to a document."""

    title: str | None = None
    author: str | None = None
    created_date: date | None = None
    language: str | None = None
    document_type: str | None = None
    product_line: str | None = None
    jurisdiction: str | None = None
    effective_date: date | None = None
    expiry_date: date | None = None
    tags: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)
