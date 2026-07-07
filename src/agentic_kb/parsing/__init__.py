"""Parser package exports.

The parsing layer normalizes many file formats into ParsedDocument objects.
Downstream chunkers should depend on these exports rather than individual
parser module internals.
"""

from agentic_kb.parsing.base import (
    Parser,
    ParserDependencyError,
    ParserLimitError,
    ParserReadError,
    ParsingLimits,
    UnsupportedContentTypeError,
)
from agentic_kb.parsing.csv import CsvParser
from agentic_kb.parsing.doc import DocParser
from agentic_kb.parsing.docx import DocxParser
from agentic_kb.parsing.html import HtmlParser
from agentic_kb.parsing.json import JsonParser
from agentic_kb.parsing.markdown import MarkdownParser
from agentic_kb.parsing.pdf import PdfParser
from agentic_kb.parsing.plain_text import PlainTextParser
from agentic_kb.parsing.registry import DOCX_CONTENT_TYPE, parser_for_path
from agentic_kb.parsing.schemas import ParsedAsset, ParsedDocument, ParsedElement, ParsedSection
from agentic_kb.parsing.xlsx import XLSX_CONTENT_TYPE, XlsxParser

__all__ = [
    "CsvParser",
    "DOCX_CONTENT_TYPE",
    "DocParser",
    "DocxParser",
    "HtmlParser",
    "JsonParser",
    "MarkdownParser",
    "ParsedAsset",
    "ParsedDocument",
    "ParsedElement",
    "ParsedSection",
    "Parser",
    "ParserDependencyError",
    "ParserLimitError",
    "ParserReadError",
    "ParsingLimits",
    "PdfParser",
    "PlainTextParser",
    "UnsupportedContentTypeError",
    "XLSX_CONTENT_TYPE",
    "XlsxParser",
    "parser_for_path",
]
