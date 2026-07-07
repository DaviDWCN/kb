"""Parser for JSON files that keeps structured fields as elements."""

import json as json_module

from agentic_kb.parsing.base import Parser
from agentic_kb.parsing.schemas import ParsedDocument, ParsedElement, ParsedSection


class JsonParser(Parser):
    """Parse JSON into readable text plus field/item-level elements."""

    supported_content_types = ("application/json", "text/json")

    def parse_text(
        self,
        text: str,
        *,
        source_uri: str,
        content_type: str,
    ) -> ParsedDocument:
        payload = json_module.loads(text)
        formatted = json_module.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
        return ParsedDocument(
            source_uri=source_uri,
            content_type=content_type,
            sections=[ParsedSection(index=0, title="JSON Document", text=formatted)],
            metadata={"root_type": _root_type(payload)},
            elements=_json_elements(payload),
        )


def _root_type(payload: object) -> str:
    if isinstance(payload, dict):
        return "object"
    if isinstance(payload, list):
        return "array"
    return type(payload).__name__


def _json_elements(payload: object) -> list[ParsedElement]:
    elements: list[ParsedElement] = []

    if isinstance(payload, dict):
        for key in sorted(payload):
            value = payload[key]
            elements.append(
                ParsedElement(
                    index=len(elements),
                    kind="json_field",
                    text=f"{key}: {json_module.dumps(value, ensure_ascii=False, sort_keys=True)}",
                    metadata={"path": key, "value_type": _root_type(value)},
                )
            )
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            elements.append(
                ParsedElement(
                    index=len(elements),
                    kind="json_item",
                    text=json_module.dumps(value, ensure_ascii=False, sort_keys=True),
                    metadata={"path": str(index), "value_type": _root_type(value)},
                )
            )
    else:
        elements.append(
            ParsedElement(
                index=0,
                kind="json_value",
                text=json_module.dumps(payload, ensure_ascii=False, sort_keys=True),
                metadata={"path": "", "value_type": _root_type(payload)},
            )
        )

    return elements
