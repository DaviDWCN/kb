"""Lightweight Markdown parser that preserves headings and paragraphs."""

from agentic_kb.parsing.base import Parser
from agentic_kb.parsing.schemas import ParsedDocument, ParsedElement, ParsedSection


class MarkdownParser(Parser):
    """Parse Markdown into heading-aware sections and ordered elements."""

    supported_content_types = ("text/markdown", "text/x-markdown")

    def parse_text(
        self,
        text: str,
        *,
        source_uri: str,
        content_type: str,
    ) -> ParsedDocument:
        front_matter, body = _extract_front_matter(text)
        sections, elements = _parse_markdown(body)
        return ParsedDocument(
            source_uri=source_uri,
            content_type=content_type,
            sections=sections,
            elements=elements,
            metadata=front_matter,
        )


def _parse_markdown_sections(text: str) -> list[ParsedSection]:
    sections, _ = _parse_markdown(text)
    return sections


def _parse_markdown(text: str) -> tuple[list[ParsedSection], list[ParsedElement]]:
    sections: list[ParsedSection] = []
    elements: list[ParsedElement] = []
    heading_stack: list[str] = []
    current_title: str | None = None
    current_path: tuple[str, ...] = ()
    current_lines: list[str] = []
    current_paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal current_paragraph_lines
        paragraph = "\n".join(current_paragraph_lines).strip()
        if paragraph:
            elements.append(
                ParsedElement(
                    index=len(elements),
                    kind="paragraph",
                    text=paragraph,
                    section_path=current_path,
                )
            )
        current_paragraph_lines = []

    for line in text.splitlines():
        heading = _parse_heading(line)
        if heading is None:
            current_lines.append(line)
            if line.strip():
                current_paragraph_lines.append(line.strip())
            else:
                flush_paragraph()
            continue

        flush_paragraph()
        if current_title is not None or "".join(current_lines).strip():
            sections.append(
                ParsedSection(
                    index=len(sections),
                    title=current_title,
                    path=current_path,
                    text="\n".join(current_lines).strip(),
                )
            )

        level, title = heading
        heading_stack = heading_stack[: level - 1]
        heading_stack.append(title)
        current_title = title
        current_path = tuple(heading_stack)
        current_lines = []
        elements.append(
            ParsedElement(
                index=len(elements),
                kind="heading",
                text=title,
                section_path=current_path,
                metadata={"level": level},
            )
        )

    flush_paragraph()
    if current_title is not None or "".join(current_lines).strip():
        sections.append(
            ParsedSection(
                index=len(sections),
                title=current_title,
                path=current_path,
                text="\n".join(current_lines).strip(),
            )
        )

    if sections:
        return sections, elements

    fallback_text = text.strip()
    if fallback_text and not elements:
        elements.append(ParsedElement(index=0, kind="paragraph", text=fallback_text))
    return [ParsedSection(index=0, text=fallback_text)], elements


import re

_FRONT_MATTER_PATTERN = re.compile(r"^>\s*\*\*(.+?)\*\*:\s*(.+)")


def _extract_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Extract `> **key**: value` metadata block from before the first heading.

    Returns (metadata_dict, remaining_body_text).
    """
    metadata: dict[str, str] = {}
    lines = text.splitlines()
    body_start = 0
    for i, line in enumerate(lines):
        match = _FRONT_MATTER_PATTERN.match(line)
        if match:
            metadata[match.group(1)] = match.group(2).strip()
        elif line.strip().startswith("#"):
            body_start = i
            break
        elif line.strip():
            # Non-front-matter, non-heading text — front-matter block ended.
            break
    return metadata, "\n".join(lines[body_start:])


def _parse_heading(line: str) -> tuple[int, str] | None:
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
