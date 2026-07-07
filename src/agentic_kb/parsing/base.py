"""Shared parser base classes, errors, and resource guardrails."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from agentic_kb.parsing.schemas import ParsedDocument


class UnsupportedContentTypeError(ValueError):
    """Raised when a parser is asked to handle an unsupported media type."""

    pass


class ParserDependencyError(RuntimeError):
    """Raised when an optional parser dependency is required but unavailable."""

    pass


class ParserLimitError(RuntimeError):
    """Raised when input or parsed output exceeds configured safety limits."""

    pass


class ParserReadError(RuntimeError):
    """Raised when content cannot be read as the requested document type."""

    pass


@dataclass(frozen=True)
class ParsingLimits:
    """Resource limits applied before and after parsing.

    These limits are soft application-level safeguards. They prevent obviously
    oversized inputs and runaway parsed structures, but process/container
    isolation is still needed for hard memory limits around heavy libraries.
    """

    max_content_bytes: int = 50 * 1024 * 1024
    max_pages: int = 500
    max_sections: int = 10_000
    max_elements: int = 100_000


class Parser(ABC):
    """Base parser that normalizes content type and enforces common limits."""

    supported_content_types: tuple[str, ...] = ()

    def __init__(self, limits: ParsingLimits | None = None) -> None:
        self.limits = limits or ParsingLimits()

    def parse(
        self,
        content: bytes | str,
        *,
        source_uri: str,
        content_type: str,
    ) -> ParsedDocument:
        """Parse bytes or text into the shared ParsedDocument contract."""

        normalized_content_type = _normalize_content_type(content_type)
        if normalized_content_type not in self.supported_content_types:
            raise UnsupportedContentTypeError(
                f"{self.__class__.__name__} does not support {content_type}"
            )
        self._validate_content_size(content)
        try:
            text = _decode_text(content)
            document = self.parse_text(
                text,
                source_uri=source_uri,
                content_type=normalized_content_type,
            )
        except (ParserDependencyError, ParserLimitError, ParserReadError, UnsupportedContentTypeError):
            raise
        except Exception as error:
            raise_parser_read_error(source_uri, normalized_content_type, error)

        self._validate_document_limits(document)
        return document

    @abstractmethod
    def parse_text(
        self,
        text: str,
        *,
        source_uri: str,
        content_type: str,
    ) -> ParsedDocument:
        raise NotImplementedError

    def _validate_content_size(self, content: bytes | str) -> None:
        """Reject content before parsing if it is larger than configured limits."""

        size = len(content.encode("utf-8")) if isinstance(content, str) else len(content)
        if size > self.limits.max_content_bytes:
            raise ParserLimitError(
                f"content size {size} bytes exceeds limit of {self.limits.max_content_bytes} bytes"
            )

    def _validate_document_limits(self, document: ParsedDocument) -> None:
        """Reject parsed output that expands into too many sections/elements."""

        if len(document.sections) > self.limits.max_sections:
            raise ParserLimitError(
                f"section count {len(document.sections)} exceeds limit of {self.limits.max_sections}"
            )
        if len(document.elements) > self.limits.max_elements:
            raise ParserLimitError(
                f"element count {len(document.elements)} exceeds limit of {self.limits.max_elements}"
            )


def _normalize_content_type(content_type: str) -> str:
    return content_type.split(";", 1)[0].strip().lower()


def _decode_text(content: bytes | str) -> str:
    if isinstance(content, str):
        return content
    return content.decode("utf-8-sig")


def raise_parser_read_error(source_uri: str, content_type: str, error: Exception) -> None:
    normalized_content_type = _normalize_content_type(content_type)
    raise ParserReadError(
        f"Could not parse {source_uri} as {normalized_content_type}: {error}"
    ) from error
