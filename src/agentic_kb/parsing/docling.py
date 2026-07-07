"""Docling-backed rich parser for complex document formats."""

from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from agentic_kb.parsing.base import (
    Parser,
    ParserDependencyError,
    ParserLimitError,
    ParserReadError,
    ParsingLimits,
    UnsupportedContentTypeError,
    raise_parser_read_error,
)
from agentic_kb.parsing.markdown import MarkdownParser
from agentic_kb.parsing.schemas import ParsedAsset, ParsedDocument, ParsedElement, ParsedSection


ConverterFactory = Callable[[], Any]
ImageCaptioner = Callable[[Any], str]
_AUTO_CONVERTER = object()


class DoclingParser(Parser):
    """Convert documents through Docling, then normalize Markdown output."""

    supported_content_types = (
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "text/html",
        "application/xhtml+xml",
        "text/markdown",
        "text/plain",
        "image/png",
        "image/jpeg",
        "image/tiff",
    )

    def __init__(
        self,
        converter_factory: ConverterFactory | None | object = _AUTO_CONVERTER,
        *,
        image_captioner: ImageCaptioner | None = None,
        limits: ParsingLimits | None = None,
    ) -> None:
        super().__init__(limits=limits)
        self._converter_factory = converter_factory
        self._image_captioner = image_captioner

    def parse_text(
        self,
        text: str,
        *,
        source_uri: str,
        content_type: str,
    ) -> ParsedDocument:
        return self.parse(
            text.encode("utf-8"),
            source_uri=source_uri,
            content_type=content_type,
        )

    def parse(
        self,
        content: bytes | str,
        *,
        source_uri: str,
        content_type: str,
    ) -> ParsedDocument:
        normalized_content_type = content_type.split(";", 1)[0].strip().lower()
        if normalized_content_type not in self.supported_content_types:
            raise UnsupportedContentTypeError(f"DoclingParser does not support {content_type}")
        self._validate_content_size(content)

        converter_factory = self._converter_factory
        if converter_factory is _AUTO_CONVERTER:
            converter_factory = _default_converter_factory()
        if converter_factory is None:
            raise ParserDependencyError(
                "Docling parsing requires the docling package. Install it to enable DoclingParser."
            )

        content_bytes = content.encode("utf-8") if isinstance(content, str) else content
        suffix = _suffix_for(source_uri, normalized_content_type)
        try:
            with NamedTemporaryFile(suffix=suffix) as temporary_file:
                temporary_file.write(content_bytes)
                temporary_file.flush()
                result = converter_factory().convert(temporary_file.name)

            native_document = getattr(result, "document", result)
            native = _parse_native_document(native_document, image_captioner=self._image_captioner)
            if native is None:
                markdown = _export_markdown(result)
                parsed = MarkdownParser().parse(
                    markdown,
                    source_uri=source_uri,
                    content_type="text/markdown",
                )
                sections = parsed.sections
                assets = parsed.assets
                elements = parsed.elements or _elements_from_markdown(markdown)
                metadata = dict(parsed.metadata)
            else:
                sections, assets, elements, metadata = native
        except (ParserDependencyError, ParserLimitError, ParserReadError, UnsupportedContentTypeError):
            raise
        except Exception as error:
            raise_parser_read_error(source_uri, normalized_content_type, error)

        document = ParsedDocument(
            source_uri=source_uri,
            content_type=normalized_content_type,
            sections=sections,
            assets=assets,
            elements=elements,
            metadata={
                **metadata,
                "parser": "docling",
                "source_format": normalized_content_type,
            },
        )
        self._validate_document_limits(document)
        return document


def _default_converter_factory() -> ConverterFactory | None:
    """Resolve Docling lazily so the project can run without the dependency."""

    try:
        from docling.document_converter import DocumentConverter

        return DocumentConverter
    except ImportError:
        return None


def _export_markdown(result: Any) -> str:
    """Get Markdown from either a Docling result or document-like object."""

    document = getattr(result, "document", result)
    export_to_markdown = getattr(document, "export_to_markdown", None)
    if callable(export_to_markdown):
        return export_to_markdown()
    raise ValueError("Docling result does not expose export_to_markdown()")


def _parse_native_document(
    document: Any,
    *,
    image_captioner: ImageCaptioner | None,
) -> tuple[list[ParsedSection], list[ParsedAsset], list[ParsedElement], dict[str, Any]] | None:
    iterate_items = getattr(document, "iterate_items", None)
    if not callable(iterate_items):
        return None

    assets: list[ParsedAsset] = []
    elements: list[ParsedElement] = []
    heading_stack: list[str] = []

    for item_index, item in enumerate(iterate_items()):
        if isinstance(item, tuple) and item:
            item = item[0]
        kind = _item_kind(item)
        if _is_heading(kind):
            text = _item_text(item)
            if not text:
                continue
            level = _heading_level(item)
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(text)
            elements.append(
                ParsedElement(
                    index=len(elements),
                    kind="heading",
                    text=text,
                    section_path=tuple(heading_stack),
                    metadata={"level": level, "item_index": item_index},
                )
            )
            continue

        if _is_table(kind):
            table = _table_element(item, len(elements), item_index, tuple(heading_stack))
            if table is not None:
                elements.append(table)
            continue

        if _is_image(kind):
            asset_id = f"image-{item_index}"
            label = getattr(item, "label", None)
            assets.append(
                ParsedAsset(
                    id=asset_id,
                    kind="image",
                    metadata={
                        "item_index": item_index,
                        **({"label": label} if isinstance(label, str) and label.strip() else {}),
                    },
                )
            )
            if image_captioner is not None:
                caption = image_captioner(item).strip()
                if caption:
                    elements.append(
                        ParsedElement(
                            index=len(elements),
                            kind="image",
                            text=caption,
                            section_path=tuple(heading_stack),
                            metadata={"asset_id": asset_id, "item_index": item_index},
                        )
                    )
            continue

        text = _item_text(item)
        if text:
            elements.append(
                ParsedElement(
                    index=len(elements),
                    kind="paragraph",
                    text=text,
                    section_path=tuple(heading_stack),
                    metadata={"item_index": item_index},
                )
            )

    return _sections_from_elements(elements), assets, elements, {"native_item_count": len(elements)}


def _table_element(
    item: Any,
    element_index: int,
    item_index: int,
    section_path: tuple[str, ...],
) -> ParsedElement | None:
    rows = [[_cell_text(cell) for cell in row] for row in getattr(item, "rows", [])]
    if not rows:
        return None

    columns = rows[0]
    data_rows = rows[1:]
    return ParsedElement(
        index=element_index,
        kind="table",
        text="\n".join(" | ".join(row) for row in rows),
        section_path=section_path,
        metadata={
            "item_index": item_index,
            "columns": columns,
            "rows": data_rows,
            "row_count": len(data_rows),
            "column_count": len(columns),
        },
    )


def _sections_from_elements(elements: list[ParsedElement]) -> list[ParsedSection]:
    sections: list[ParsedSection] = []
    current_title: str | None = None
    current_path: tuple[str, ...] = ()
    current_lines: list[str] = []

    for element in elements:
        if element.kind == "heading":
            if current_title is not None or current_lines:
                sections.append(
                    ParsedSection(
                        index=len(sections),
                        title=current_title,
                        path=current_path,
                        text="\n".join(current_lines).strip(),
                    )
                )
            current_title = element.text
            current_path = element.section_path
            current_lines = []
            continue

        current_lines.append(element.text)

    if current_title is not None or current_lines:
        sections.append(
            ParsedSection(
                index=len(sections),
                title=current_title,
                path=current_path,
                text="\n".join(current_lines).strip(),
            )
        )
    return sections or [ParsedSection(index=0, title="Docling Document", text="")]


def _item_kind(item: Any) -> str:
    value = getattr(item, "kind", None)
    if value is not None:
        return str(value).lower()
    return item.__class__.__name__.lower()


def _item_text(item: Any) -> str:
    value = getattr(item, "text", None)
    if isinstance(value, str):
        return value.strip()
    return ""


def _heading_level(item: Any) -> int:
    level = getattr(item, "level", None)
    return level if isinstance(level, int) and 1 <= level <= 6 else 1


def _is_heading(kind: str) -> bool:
    return "heading" in kind or kind in {"section_header", "title"}


def _is_table(kind: str) -> bool:
    return "table" in kind


def _is_image(kind: str) -> bool:
    return any(marker in kind for marker in ("image", "picture", "figure"))


def _cell_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _elements_from_markdown(markdown: str) -> list[ParsedElement]:
    """Fallback Markdown-to-element conversion for Docling outputs."""

    elements: list[ParsedElement] = []
    heading_stack: list[str] = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        text = "\n".join(paragraph_lines).strip()
        if text:
            elements.append(
                ParsedElement(
                    index=len(elements),
                    kind="paragraph",
                    text=text,
                    section_path=tuple(heading_stack),
                )
            )
        paragraph_lines = []

    for line in markdown.splitlines():
        heading = _parse_markdown_heading(line)
        if heading is not None:
            flush_paragraph()
            level, title = heading
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)
            elements.append(
                ParsedElement(
                    index=len(elements),
                    kind="heading",
                    text=title,
                    section_path=tuple(heading_stack),
                    metadata={"level": level},
                )
            )
            continue
        if line.strip():
            paragraph_lines.append(line.strip())
        else:
            flush_paragraph()

    flush_paragraph()
    return elements


def _parse_markdown_heading(line: str) -> tuple[int, str] | None:
    """Return heading level/title when a Markdown line is an ATX heading."""

    stripped = line.strip()
    if not stripped.startswith("#"):
        return None
    marker, _, title = stripped.partition(" ")
    if not title or any(character != "#" for character in marker):
        return None
    level = len(marker)
    if level > 6:
        return None
    return level, title.strip()


def _suffix_for(source_uri: str, content_type: str) -> str:
    """Choose a temporary filename suffix so Docling can infer format."""

    suffix = Path(source_uri).suffix
    if suffix:
        return suffix
    return {
        "application/pdf": ".pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
        "text/html": ".html",
        "application/xhtml+xml": ".xhtml",
        "text/markdown": ".md",
        "text/plain": ".txt",
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/tiff": ".tiff",
    }.get(content_type, "")
