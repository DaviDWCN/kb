"""HTML cleanup parser that preserves headings and paragraph blocks."""

from bs4 import BeautifulSoup

from agentic_kb.parsing.base import Parser
from agentic_kb.parsing.schemas import ParsedDocument, ParsedElement, ParsedSection


_IGNORED_TAGS = {"script", "style", "nav", "noscript"}
_BLOCK_TAGS = {"p", "li", "td", "th", "blockquote", "pre"}
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_CONTENT_TAGS = tuple(sorted(_BLOCK_TAGS | _HEADING_TAGS))


class HtmlParser(Parser):
    """Parse HTML while ignoring script/style/navigation noise."""

    supported_content_types = ("text/html", "application/xhtml+xml")

    def parse_text(
        self,
        text: str,
        *,
        source_uri: str,
        content_type: str,
    ) -> ParsedDocument:
        blocks = _content_blocks(text)
        sections, elements = _document_parts_from_blocks(blocks)
        return ParsedDocument(
            source_uri=source_uri,
            content_type=content_type,
            sections=sections,
            elements=elements,
        )


def _content_blocks(text: str) -> list[tuple[str, int | None, str]]:
    """Extract ordered heading and text blocks with BeautifulSoup."""

    soup = BeautifulSoup(text, "html.parser")
    for tag in soup.find_all(_IGNORED_TAGS):
        tag.decompose()

    blocks: list[tuple[str, int | None, str]] = []
    for tag in soup.find_all(_CONTENT_TAGS):
        block_text = " ".join(tag.get_text(" ", strip=True).split())
        if not block_text:
            continue
        if tag.name in _HEADING_TAGS:
            blocks.append(("heading", int(tag.name[1]), block_text))
            continue
        blocks.append(("block", None, block_text))
    return blocks


def _sections_from_blocks(blocks: list[tuple[str, int | None, str]]) -> list[ParsedSection]:
    sections, _ = _document_parts_from_blocks(blocks)
    return sections


def _document_parts_from_blocks(
    blocks: list[tuple[str, int | None, str]]
) -> tuple[list[ParsedSection], list[ParsedElement]]:
    sections: list[ParsedSection] = []
    elements: list[ParsedElement] = []
    heading_stack: list[str] = []
    current_title: str | None = None
    current_path: tuple[str, ...] = ()
    current_lines: list[str] = []

    for kind, level, text in blocks:
        if kind != "heading":
            current_lines.append(text)
            elements.append(
                ParsedElement(
                    index=len(elements),
                    kind="paragraph",
                    text=text,
                    section_path=current_path,
                )
            )
            continue

        if current_title is not None or current_lines:
            sections.append(
                ParsedSection(
                    index=len(sections),
                    title=current_title,
                    path=current_path,
                    text="\n".join(current_lines).strip(),
                )
            )

        heading_level = level or 1
        heading_stack = heading_stack[: heading_level - 1]
        heading_stack.append(text)
        current_title = text
        current_path = tuple(heading_stack)
        current_lines = []
        elements.append(
            ParsedElement(
                index=len(elements),
                kind="heading",
                text=text,
                section_path=current_path,
                metadata={"level": heading_level},
            )
        )

    if current_title is not None or current_lines:
        sections.append(
            ParsedSection(
                index=len(sections),
                title=current_title,
                path=current_path,
                text="\n".join(current_lines).strip(),
            )
        )

    return sections or [ParsedSection(index=0, text="")], elements
