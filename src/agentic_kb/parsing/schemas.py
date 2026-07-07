"""Normalized parser output contracts consumed by chunking."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParsedAsset:
    """Non-text artifact discovered while parsing, such as an image or table file."""

    id: str
    kind: str
    metadata: dict[str, Any] = field(default_factory=dict)
    content_type: str | None = None
    source_uri: str | None = None


@dataclass(frozen=True)
class ParsedSection:
    """Logical document section used for human inspection and fallback chunking."""

    index: int
    text: str
    title: str | None = None
    path: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedElement:
    """Ordered source-level element with clean text and structural metadata."""

    index: int
    kind: str
    text: str
    section_path: tuple[str, ...] = ()
    page_number: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    start_char: int | None = None
    end_char: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedDocument:
    """Canonical parser output shared by all parser implementations."""

    source_uri: str
    content_type: str
    sections: list[ParsedSection]
    metadata: dict[str, Any] = field(default_factory=dict)
    assets: list[ParsedAsset] = field(default_factory=list)
    elements: list[ParsedElement] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(section.text for section in self.sections if section.text)
