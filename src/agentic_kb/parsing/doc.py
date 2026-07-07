""".doc (legacy Word binary) parser using pywin32 / Word COM.

Converts .doc → .docx via Word, then delegates to DocxParser for text
extraction and optional Qwen VL image description.
"""

import os
import tempfile
from collections.abc import Callable

from agentic_kb.parsing.base import (
    Parser,
    ParserLimitError,
    ParserReadError,
    ParsingLimits,
    UnsupportedContentTypeError,
    raise_parser_read_error,
)
from agentic_kb.parsing.schemas import ParsedDocument, ParsedElement, ParsedSection

DOC_CONTENT_TYPE = "application/msword"
ImageDescriber = Callable[[bytes, str], str]


class DocParser(Parser):
    """Extract text from legacy .doc files via Microsoft Word COM.

    Converts .doc → .docx in-memory, then delegates to DocxParser so
    images are described via the same Qwen VL pipeline used for .docx.
    """

    supported_content_types = (DOC_CONTENT_TYPE,)

    def __init__(
        self,
        *,
        image_describer: ImageDescriber | None = None,
        limits: ParsingLimits | None = None,
    ) -> None:
        super().__init__(limits=limits)
        self._image_describer = image_describer

    def parse_text(self, text: str, *, source_uri: str, content_type: str) -> ParsedDocument:
        raise TypeError("DocParser requires bytes content")

    def parse(self, content: bytes | str, *, source_uri: str, content_type: str) -> ParsedDocument:
        if isinstance(content, str):
            content = content.encode("utf-8")
        self._ensure_supported(content_type)
        self._validate_content_size(content)

        docx_bytes = _convert_doc_to_docx(content)
        if docx_bytes:
            try:
                from agentic_kb.parsing.docx import DocxParser

                docx_parser = DocxParser(image_describer=self._image_describer)
                return docx_parser.parse(
                    docx_bytes,
                    source_uri=source_uri,
                    content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            except Exception:
                pass

        # Fallback: plain text extraction via Word COM
        text = _extract_doc_text(content)
        text = text.replace("\x07", "")
        if not text.strip():
            text = "(no extractable text from .doc file)"

        return ParsedDocument(
            source_uri=source_uri,
            content_type=DOC_CONTENT_TYPE,
            sections=[ParsedSection(index=0, title="Word Document", text=text)],
            elements=[ParsedElement(index=0, kind="paragraph", text=text)],
        )

    def _ensure_supported(self, content_type: str) -> None:
        normalized = content_type.split(";", 1)[0].strip().lower()
        if normalized not in self.supported_content_types:
            raise UnsupportedContentTypeError(f"DocParser does not support {content_type}")


# ---------------------------------------------------------------------------
# Word COM helpers
# ---------------------------------------------------------------------------

def _word_com_open(content: bytes, suffix: str):
    """Open a Word COM session, write content to temp file, open in Word.

    Returns (word_app, document, temp_path) or raises.
    """
    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False

    fd, doc_path = tempfile.mkstemp(suffix=suffix)
    os.write(fd, content)
    os.close(fd)

    doc = word.Documents.Open(doc_path)
    return word, doc, doc_path


def _word_com_close(word, doc, doc_path: str | None) -> None:
    """Safely close a Word COM session and clean up temp files."""
    import pythoncom

    try:
        doc.Close()
    except Exception:
        pass
    try:
        word.Quit()
    except Exception:
        pass
    try:
        pythoncom.CoUninitialize()
    except Exception:
        pass
    if doc_path and os.path.exists(doc_path):
        try:
            os.unlink(doc_path)
        except OSError:
            pass


def _convert_doc_to_docx(content: bytes) -> bytes | None:
    """Use Word COM to convert .doc → .docx in memory.  Returns None on failure."""
    try:
        import pythoncom
        import win32com.client
    except ImportError:
        return None

    word = None
    doc = None
    doc_path: str | None = None
    docx_path: str | None = None
    try:
        word, doc, doc_path = _word_com_open(content, suffix=".doc")

        fd, docx_path = tempfile.mkstemp(suffix=".docx")
        os.close(fd)

        # 16 = wdFormatXMLDocument (.docx)
        doc.SaveAs2(docx_path, FileFormat=16)
        docx_bytes = open(docx_path, "rb").read()
        return docx_bytes
    except Exception:
        return None
    finally:
        if doc is not None:
            _word_com_close(word, doc, doc_path)
        if docx_path and os.path.exists(docx_path):
            try:
                os.unlink(docx_path)
            except OSError:
                pass


def _extract_doc_text(content: bytes) -> str:
    """Extract plain text from .doc via Word COM.  Returns '' on failure."""
    try:
        import pythoncom
        import win32com.client
    except ImportError:
        return ""

    word = None
    doc = None
    doc_path: str | None = None
    try:
        word, doc, doc_path = _word_com_open(content, suffix=".doc")
        text = doc.Content.Text
        return text.strip()
    except Exception:
        return ""
    finally:
        if doc is not None:
            _word_com_close(word, doc, doc_path)