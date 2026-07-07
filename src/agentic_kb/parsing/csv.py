"""Parser for CSV data that preserves table and row structure."""

import csv
from io import StringIO

from agentic_kb.parsing.base import Parser
from agentic_kb.parsing.schemas import ParsedDocument, ParsedElement, ParsedSection


class CsvParser(Parser):
    """Parse CSV into table text, table metadata, and row elements."""

    supported_content_types = ("text/csv", "application/csv")

    def parse_text(
        self,
        text: str,
        *,
        source_uri: str,
        content_type: str,
    ) -> ParsedDocument:
        rows = list(csv.reader(StringIO(text)))
        if not rows:
            table_text = ""
            columns: list[str] = []
            row_count = 0
            elements: list[ParsedElement] = []
        else:
            columns = rows[0]
            data_rows = rows[1:]
            table_text = "\n".join(" | ".join(cell for cell in row) for row in rows)
            row_count = len(data_rows)
            elements = [
                ParsedElement(
                    index=0,
                    kind="table",
                    text=table_text,
                    metadata={"row_count": row_count, "columns": columns},
                )
            ]
            for row_index, row in enumerate(data_rows):
                elements.append(
                    ParsedElement(
                        index=len(elements),
                        kind="row",
                        text=" | ".join(row),
                        metadata={
                            "row_index": row_index,
                            "values": dict(zip(columns, row)),
                        },
                    )
                )

        return ParsedDocument(
            source_uri=source_uri,
            content_type=content_type,
            sections=[ParsedSection(index=0, title="CSV Table", text=table_text)],
            metadata={"row_count": row_count, "columns": columns},
            elements=elements,
        )
