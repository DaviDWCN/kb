"""PDF parser with optional pypdf/PyPDF2 support and page limits."""

from collections.abc import Callable
from io import BytesIO
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
from agentic_kb.parsing.schemas import ParsedDocument, ParsedElement, ParsedSection


PdfReaderFactory = Callable[[BytesIO], Any]
OcrTextExtractor = Callable[[bytes, int], str]
_AUTO_READER = object()
_AUTO_OCR = object()
_DEFAULT_MIN_NATIVE_TEXT_CHARS = 8


class PdfParser(Parser):
    """Extract each PDF page as a section and page-level element."""

    supported_content_types = ("application/pdf",)

    def __init__(
        self,
        reader_factory: PdfReaderFactory | None | object = _AUTO_READER,
        *,
        ocr_text_extractor: OcrTextExtractor | None | object = _AUTO_OCR,
        min_native_text_chars: int = _DEFAULT_MIN_NATIVE_TEXT_CHARS,
        limits: ParsingLimits | None = None,
    ) -> None:
        super().__init__(limits=limits)
        self._reader_factory = reader_factory
        self._ocr_text_extractor = ocr_text_extractor
        self._min_native_text_chars = min_native_text_chars

    def parse_text(
        self,
        text: str,
        *,
        source_uri: str,
        content_type: str,
    ) -> ParsedDocument:
        raise TypeError("PdfParser requires bytes content")

    def parse(
        self,
        content: bytes | str,
        *,
        source_uri: str,
        content_type: str,
    ) -> ParsedDocument:
        if isinstance(content, str):
            content = content.encode("utf-8")
        self._ensure_supported(content_type)
        self._validate_content_size(content)
        normalized_content_type = content_type.split(";", 1)[0].strip().lower()
        reader_factory = self._reader_factory
        if reader_factory is _AUTO_READER:
            reader_factory = _default_reader_factory()
        if reader_factory is None:
            raise ParserDependencyError(
                "PDF parsing requires pypdf or PyPDF2. Install one of them to enable PdfParser."
            )
        ocr_text_extractor = self._ocr_text_extractor
        if ocr_text_extractor is _AUTO_OCR:
            ocr_text_extractor = _default_ocr_text_extractor()

        try:
            reader = reader_factory(BytesIO(content))
            pages = list(reader.pages)
        except (ParserDependencyError, ParserLimitError, ParserReadError, UnsupportedContentTypeError):
            raise
        except Exception as error:
            raise_parser_read_error(source_uri, normalized_content_type, error)

        if len(pages) > self.limits.max_pages:
            raise ParserLimitError(
                f"page count {len(pages)} exceeds limit of {self.limits.max_pages}"
            )
        try:
            sections = [
                _section_from_page(
                    page,
                    content=content,
                    index=index,
                    ocr_text_extractor=ocr_text_extractor,
                    min_native_text_chars=self._min_native_text_chars,
                )
                for index, page in enumerate(pages)
            ]
        except Exception as error:
            raise_parser_read_error(source_uri, normalized_content_type, error)

        elements = [
            ParsedElement(
                index=section.index,
                kind="page",
                text=section.text,
                metadata={
                    "title": section.title,
                    "extraction_method": section.metadata["extraction_method"],
                    "ocr_attempted": section.metadata["ocr_attempted"],
                },
            )
            for section in sections
        ]
        document = ParsedDocument(
            source_uri=source_uri,
            content_type=normalized_content_type,
            sections=sections,
            elements=elements,
            metadata={},
        )
        self._validate_document_limits(document)
        return document

    def _ensure_supported(self, content_type: str) -> None:
        normalized = content_type.split(";", 1)[0].strip().lower()
        if normalized not in self.supported_content_types:
            raise UnsupportedContentTypeError(f"PdfParser does not support {content_type}")


def _default_reader_factory() -> PdfReaderFactory | None:
    """Resolve an installed PDF reader lazily so the dependency stays optional."""

    try:
        from pypdf import PdfReader

        return PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader

            return PdfReader
        except ImportError:
            return None


def _default_ocr_text_extractor() -> OcrTextExtractor | None:
    """Resolve local OCR dependencies lazily when they are installed."""

    try:
        import fitz
        from PIL import Image
        import pytesseract
    except ImportError:
        return None

    def extract(content: bytes, page_number: int) -> str:
        document = fitz.open(stream=content, filetype="pdf")
        try:
            page = document.load_page(page_number - 1)
            pixmap = page.get_pixmap()
            mode = "RGBA" if pixmap.alpha else "RGB"
            image = Image.frombytes(mode, (pixmap.width, pixmap.height), pixmap.samples)
            return pytesseract.image_to_string(image, lang='chi_sim+eng').strip()
        finally:
            document.close()

    return extract


def _section_from_page(
    page: Any,
    *,
    content: bytes,
    index: int,
    ocr_text_extractor: OcrTextExtractor | None,
    min_native_text_chars: int,
) -> ParsedSection:
    text = (page.extract_text() or "").strip()
    extraction_method = "text"
    ocr_attempted = False
    if _should_attempt_ocr(text, min_native_text_chars) and ocr_text_extractor is not None:
        ocr_attempted = True
        ocr_text = (ocr_text_extractor(content, index + 1) or "").strip()
        if _use_ocr_text(native_text=text, ocr_text=ocr_text):
            text = ocr_text
            extraction_method = "ocr"

    return ParsedSection(
        index=index,
        title=f"Section {index + 1}",
        text=text,
        metadata={
            "extraction_method": extraction_method,
            "ocr_attempted": ocr_attempted,
        },
    )


def _should_attempt_ocr(text: str, min_native_text_chars: int) -> bool:
    return len(_compact_text(text)) < min_native_text_chars


def _use_ocr_text(*, native_text: str, ocr_text: str) -> bool:
    return bool(ocr_text) and len(_compact_text(ocr_text)) > len(_compact_text(native_text))


def _compact_text(text: str) -> str:
    return "".join(text.split())
