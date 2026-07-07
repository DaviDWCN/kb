"""Parser for plain text documents."""

import re

from agentic_kb.parsing.base import Parser
from agentic_kb.parsing.schemas import ParsedDocument, ParsedElement, ParsedSection


class PlainTextParser(Parser):
    """Parse plain text into one section and paragraph-level elements."""

    supported_content_types = ("text/plain",)

    def parse_text(
        self,
        text: str,
        *,
        source_uri: str,
        content_type: str,
    ) -> ParsedDocument:
        elements = _paragraph_elements(text)
        return ParsedDocument(
            source_uri=source_uri,
            content_type=content_type,
            sections=[ParsedSection(index=0, text=text)],
            elements=elements,
        )


def _paragraph_elements(text: str) -> list[ParsedElement]:
    elements: list[ParsedElement] = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, flags=re.DOTALL):
        paragraph = match.group(0).strip()
        if not paragraph:
            continue
        elements.append(
            ParsedElement(
                index=len(elements),
                kind="paragraph",
                text=paragraph,
                start_char=match.start(),
                end_char=match.start() + len(match.group(0).rstrip()),
            )
        )
    return elements
