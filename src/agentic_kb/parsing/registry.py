"""Parser lookup helpers for file-based CLI workflows."""

from pathlib import Path

from agentic_kb.parsing.base import Parser
from agentic_kb.parsing.csv import CsvParser
from agentic_kb.parsing.doc import DocParser, DOC_CONTENT_TYPE as DOC_CT
from agentic_kb.parsing.docx import DocxParser
from agentic_kb.parsing.html import HtmlParser
from agentic_kb.parsing.json import JsonParser
from agentic_kb.parsing.markdown import MarkdownParser
from agentic_kb.parsing.pdf import PdfParser
from agentic_kb.parsing.plain_text import PlainTextParser
from agentic_kb.parsing.xlsx import XLSX_CONTENT_TYPE, XlsxParser


DOCX_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PPTX_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


def parser_for_path(path: str | Path) -> tuple[Parser | None, str | None]:
    """Return the parser and content type associated with a file suffix."""
    suffix = Path(path).suffix.lower()
    if suffix in {".txt", ".text"}:
        return PlainTextParser(), "text/plain"
    if suffix in {".md", ".markdown"}:
        return MarkdownParser(), "text/markdown"
    if suffix in {".html", ".htm"}:
        return HtmlParser(), "text/html"
    if suffix == ".json":
        return JsonParser(), "application/json"
    if suffix == ".csv":
        return CsvParser(), "text/csv"
    if suffix == ".pdf":
        # Docling requires VC Redist + torch — fall back to PdfParser until DLLs are installed.
        return PdfParser(), "application/pdf"
    if suffix in {".ppt", ".pptx"}:
        return None, None  # Docling required but DLLs missing
    if suffix == ".doc":
        return DocParser(), DOC_CT
    if suffix == ".docx":
        return DocxParser(), DOCX_CONTENT_TYPE
    if suffix == ".xlsx":
        return XlsxParser(), XLSX_CONTENT_TYPE
    return None, None
