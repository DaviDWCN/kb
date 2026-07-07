"""Structure-aware recursive chunking for parsed documents."""

import re
from dataclasses import dataclass
from datetime import date, datetime

from agentic_kb.chunking.schemas import ChunkingConfig, ChunkingResult
from agentic_kb.parsing.schemas import ParsedDocument, ParsedElement, ParsedSection
from agentic_kb.schemas.chunks import Chunk, ChunkMetadata


@dataclass(frozen=True)
class _ChunkSource:
    """Internal chunking input after sections/elements are grouped."""

    text: str
    heading_path: tuple[str, ...]
    page_number: int | None
    attributes: dict[str, object]
    atomic: bool = False


_ATOMIC_ELEMENT_KINDS = {"row", "table"}
_DOCUMENT_METADATA_ATTRIBUTE_KEYS = {
    "effective_date",
    "expiry_date",
    "metadata_source",
    "metadata_confidence",
}


class RecursiveChunker:
    """Chunk ParsedDocument text while preserving source structure metadata."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = _effective_config(config or ChunkingConfig(strategy="recursive"))
        _validate_config(self.config)

    def chunk(self, document: ParsedDocument, *, document_id: str) -> ChunkingResult:
        """Convert a parsed document into deterministic Chunk objects."""

        chunks: list[Chunk] = []
        document_attributes = _filterable_document_metadata(document.metadata)
        sources = _sources_from_document(document, document_attributes=document_attributes)
        if self.config.merge_target_tokens is not None:
            sources = _pack_sources(
                sources,
                max_tokens=self.config.max_tokens,
                merge_target_tokens=self.config.merge_target_tokens,
                overlap_tokens=self.config.overlap_tokens,
            )
        for source in sources:
            for text in _split_text(
                source.text,
                self.config.max_tokens,
                self.config.overlap_tokens,
                self.config.min_tokens or 1,
            ):
                chunks.append(
                    Chunk(
                        id=f"{document_id}:chunk-{len(chunks)}",
                        document_id=document_id,
                        text=text,
                        index=len(chunks),
                        metadata=ChunkMetadata(
                            heading_path=source.heading_path,
                            page_number=source.page_number,
                            token_count=_token_count(text),
                            attributes={
                                "source_uri": document.source_uri,
                                "content_type": document.content_type,
                                **source.attributes,
                            },
                        ),
                    )
                )

        return ChunkingResult(document_id=document_id, chunks=chunks, config=self.config)


def _validate_config(config: ChunkingConfig) -> None:
    if config.max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")
    if config.overlap_tokens < 0 or config.overlap_tokens >= config.max_tokens:
        raise ValueError("overlap_tokens must be non-negative and smaller than max_tokens")
    if config.min_tokens is not None and config.min_tokens > config.max_tokens:
        raise ValueError("min_tokens must be less than or equal to max_tokens")
    if config.merge_target_tokens is not None:
        if config.merge_target_tokens <= 0:
            raise ValueError("merge_target_tokens must be greater than zero")
        if config.merge_target_tokens > config.max_tokens:
            raise ValueError("merge_target_tokens must be less than or equal to max_tokens")


def _effective_config(config: ChunkingConfig) -> ChunkingConfig:
    max_tokens = config.max_tokens
    if config.context_window_tokens is not None and config.target_chunks_per_query is not None:
        available_tokens = (
            config.context_window_tokens
            - config.reserved_prompt_tokens
            - config.reserved_answer_tokens
        )
        if available_tokens <= 0:
            raise ValueError("context token budget must be greater than zero")
        if config.target_chunks_per_query <= 0:
            raise ValueError("target_chunks_per_query must be greater than zero")
        max_tokens = min(max_tokens, max(1, available_tokens // config.target_chunks_per_query))

    min_tokens = config.min_tokens
    if min_tokens is None:
        min_tokens = max(1, max_tokens // 2)

    return ChunkingConfig(
        strategy=config.strategy,
        max_tokens=max_tokens,
        overlap_tokens=config.overlap_tokens,
        min_tokens=min_tokens,
        merge_target_tokens=config.merge_target_tokens,
        context_window_tokens=config.context_window_tokens,
        reserved_prompt_tokens=config.reserved_prompt_tokens,
        reserved_answer_tokens=config.reserved_answer_tokens,
        target_chunks_per_query=config.target_chunks_per_query,
    )


def _sources_from_document(
    document: ParsedDocument,
    *,
    document_attributes: dict[str, object] | None = None,
) -> list[_ChunkSource]:
    document_attributes = document_attributes or {}
    if document.elements:
        return _with_document_attributes(
            _sources_from_elements(document.elements),
            document_attributes,
        )
    return _with_document_attributes(
        [_source_from_section(section) for section in document.sections if section.text.strip()],
        document_attributes,
    )


def _with_document_attributes(
    sources: list[_ChunkSource],
    document_attributes: dict[str, object],
) -> list[_ChunkSource]:
    if not document_attributes:
        return sources
    return [
        _ChunkSource(
            text=source.text,
            heading_path=source.heading_path,
            page_number=source.page_number,
            attributes={**source.attributes, **document_attributes},
            atomic=source.atomic,
        )
        for source in sources
    ]


def _filterable_document_metadata(metadata: dict[str, object]) -> dict[str, object]:
    return {
        key: _metadata_value(value)
        for key, value in metadata.items()
        if key in _DOCUMENT_METADATA_ATTRIBUTE_KEYS and value is not None
    }


def _metadata_value(value: object) -> object:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def _sources_from_elements(elements: list[ParsedElement]) -> list[_ChunkSource]:
    """Group adjacent compatible elements before token splitting.

    This avoids one-chunk-per-line behavior for formats such as resumes, while
    still keeping page/table/row-like elements atomic for citation fidelity.
    """

    sources: list[_ChunkSource] = []
    current: list[ParsedElement] = []

    def flush_current() -> None:
        nonlocal current
        if current:
            sources.append(_source_from_element_group(current))
            current = []

    for element in elements:
        if not element.text.strip():
            continue
        if element.kind in _ATOMIC_ELEMENT_KINDS:
            flush_current()
            sources.append(_source_from_element_group([element]))
            continue
        if current and not _can_group_elements(current[-1], element):
            flush_current()
        current.append(element)

    flush_current()
    return sources


def _source_from_section(section: ParsedSection) -> _ChunkSource:
    page_number = section.metadata.get("page_number")
    if not isinstance(page_number, int):
        page_number = None
    return _ChunkSource(
        text=section.text.strip(),
        heading_path=section.path,
        page_number=page_number,
        attributes={
            "section_index": section.index,
            "section_title": section.title,
            **section.metadata,
        },
    )


def _source_from_element_group(elements: list[ParsedElement]) -> _ChunkSource:
    first = elements[0]
    metadata = _element_group_metadata(elements)
    return _ChunkSource(
        text="\n\n".join(element.text.strip() for element in elements if element.text.strip()),
        heading_path=first.section_path,
        page_number=_group_page_number(elements),
        attributes=metadata,
        atomic=any(element.kind in _ATOMIC_ELEMENT_KINDS for element in elements),
    )


def _pack_sources(
    sources: list[_ChunkSource],
    *,
    max_tokens: int,
    merge_target_tokens: int,
    overlap_tokens: int,
) -> list[_ChunkSource]:
    """Merge adjacent compatible sources until the target size is reached."""

    packed: list[_ChunkSource] = []
    current: list[_ChunkSource] = []

    def flush_current() -> None:
        nonlocal current
        if current:
            packed.append(_merge_sources(current))
            current = []

    for source in sources:
        if source.atomic:
            flush_current()
            packed.append(source)
            continue

        if current and (
            _token_count(_join_source_text(current)) >= merge_target_tokens
            or not _can_pack_sources(current[-1], source)
            or _token_count(_join_source_text([*current, source])) > max_tokens
        ):
            flush_current()
            current = _overlap_sources(packed[-1], overlap_tokens) if overlap_tokens else []
            if current and (
                not _can_pack_sources(current[-1], source)
                or _token_count(_join_source_text([*current, source])) > max_tokens
            ):
                current = []

        current.append(source)

    flush_current()
    return packed


def _merge_sources(sources: list[_ChunkSource]) -> _ChunkSource:
    if len(sources) == 1:
        return sources[0]

    return _ChunkSource(
        text=_join_source_text(sources),
        heading_path=sources[0].heading_path,
        page_number=_source_group_page_number(sources),
        attributes=_source_group_metadata(sources),
    )


def _join_source_text(sources: list[_ChunkSource]) -> str:
    return "\n\n".join(source.text.strip() for source in sources if source.text.strip())


def _can_pack_sources(left: _ChunkSource, right: _ChunkSource) -> bool:
    if left.atomic or right.atomic:
        return False
    return True


def _overlap_sources(previous: _ChunkSource, overlap_tokens: int) -> list[_ChunkSource]:
    if overlap_tokens <= 0:
        return []

    overlap_text = _last_tokens(previous.text, overlap_tokens)
    if not overlap_text:
        return []

    return [
        _ChunkSource(
            text=overlap_text,
            heading_path=previous.heading_path,
            page_number=previous.page_number,
            attributes={"is_overlap": True, "overlap_from": previous.attributes},
        )
    ]


def _last_tokens(text: str, token_count: int) -> str:
    tokens = text.split()
    if len(tokens) <= 1:
        # CJK text without spaces: use last N characters
        return text[-token_count:] if token_count > 0 else ""
    return " ".join(tokens[-token_count:]) if tokens else ""


def _source_group_page_number(sources: list[_ChunkSource]) -> int | None:
    page_numbers = {source.page_number for source in sources if source.page_number is not None}
    if len(page_numbers) == 1:
        return next(iter(page_numbers))
    return None


def _source_group_metadata(sources: list[_ChunkSource]) -> dict[str, object]:
    metadata: dict[str, object] = {}
    metadata.update(_common_document_metadata(sources))

    section_indices = [
        source.attributes["section_index"]
        for source in sources
        if "section_index" in source.attributes
    ]
    if section_indices:
        metadata["section_indices"] = tuple(section_indices)

    return metadata


def _common_document_metadata(sources: list[_ChunkSource]) -> dict[str, object]:
    common: dict[str, object] = {}
    for key in _DOCUMENT_METADATA_ATTRIBUTE_KEYS:
        values = [
            source.attributes[key]
            for source in sources
            if key in source.attributes
        ]
        if len(values) == len(sources) and all(value == values[0] for value in values):
            common[key] = values[0]
    return common


def _split_text(text: str, max_tokens: int, overlap_tokens: int, min_tokens: int) -> list[str]:
    """Split text by sentence-like units, falling back to token windows."""

    if _token_count(text) <= max_tokens:
        return [text.strip()] if text.strip() else []

    units = _sentence_units(text)
    chunks: list[str] = []
    current_units: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = _token_count(unit)
        if unit_tokens > max_tokens:
            if current_units:
                chunks.append(" ".join(current_units).strip())
                current_units = []
                current_tokens = 0
            chunks.extend(_split_tokens(unit, max_tokens, overlap_tokens, min_tokens))
            continue

        if current_units and current_tokens + unit_tokens > max_tokens:
            chunks.append(" ".join(current_units).strip())
            current_units = _overlap_units(current_units, overlap_tokens)
            current_tokens = sum(_token_count(value) for value in current_units)

        current_units.append(unit)
        current_tokens += unit_tokens

    if current_units:
        chunks.append(" ".join(current_units).strip())

    return _merge_small_tail([chunk for chunk in chunks if chunk], max_tokens, min_tokens)


def _sentence_units(text: str) -> list[str]:
    # Split on line breaks first (single or double)
    lines = [line.strip() for line in re.split(r"\n+", text) if line.strip()]
    if not lines:
        return []
    units: list[str] = []
    for line in lines:
        # Split on English AND Chinese sentence endings
        sentences = re.split(r"(?<=[.!?\u3002\uff01\uff1f])\s*", line)
        units.extend(sentence.strip() for sentence in sentences if sentence.strip())
    return units or [text.strip()]


def _split_tokens(text: str, max_tokens: int, overlap_tokens: int, min_tokens: int) -> list[str]:
    tokens = text.split()
    # Fallback: split by characters for CJK text or any text with no whitespace
    if len(tokens) <= 1 and len(text) > max_tokens:
        tokens = list(text)
    chunks: list[str] = []
    start = 0
    step = max_tokens - overlap_tokens
    while start < len(tokens):
        if start > 0 and len(tokens) - start <= overlap_tokens:
            break
        # Build chunk by accumulating until estimated tokens exceed max_tokens
        chunk_tokens: list[str] = []
        chunk_est = 0
        j = start
        while j < len(tokens) and chunk_est + _token_count(tokens[j]) <= max_tokens:
            chunk_tokens.append(tokens[j])
            chunk_est += _token_count(tokens[j])
            j += 1
        if not chunk_tokens and j < len(tokens):
            chunk_tokens = [tokens[j]]
        # Join: characters for CJK mode, spaces for word mode
        sep = "" if (len(tokens) == len(text)) else " "
        chunks.append(sep.join(chunk_tokens))
        start += step
    return _merge_small_tail(chunks, max_tokens, min_tokens)


def _overlap_units(units: list[str], overlap_tokens: int) -> list[str]:
    if overlap_tokens == 0:
        return []

    selected: list[str] = []
    selected_tokens = 0
    for unit in reversed(units):
        unit_tokens = _token_count(unit)
        if selected and selected_tokens + unit_tokens > overlap_tokens:
            break
        selected.insert(0, unit)
        selected_tokens += unit_tokens
        if selected_tokens >= overlap_tokens:
            break
    return selected


def _token_count(text: str) -> int:
    """Estimate token count by characters (~1 tok per CJK char, ~4 chars per English tok)."""
    import re
    cjk = len(re.findall(r'[\u4e00-\u9fff]', text))
    other = max(len(text) - cjk, 0)
    return cjk + (other + 3) // 4  # ~1 token per CJK char, ~4 chars per token for English


def _merge_small_tail(chunks: list[str], max_tokens: int, min_tokens: int) -> list[str]:
    """Merge consecutive small chunks until they reach min_tokens, without exceeding max_tokens."""
    if len(chunks) < 2:
        return chunks

    result: list[str] = []
    i = 0
    while i < len(chunks):
        current_tokens = _token_count(chunks[i])

        if current_tokens >= min_tokens:
            result.append(chunks[i])
            i += 1
            continue

        # Accumulate consecutive small chunks while staying within max_tokens
        merged_parts = [chunks[i]]
        merged_tokens = current_tokens
        j = i + 1

        while j < len(chunks):
            next_tokens = _token_count(chunks[j])
            if merged_tokens + next_tokens > max_tokens:
                break
            merged_parts.append(chunks[j])
            merged_tokens += next_tokens
            j += 1

        if len(merged_parts) > 1:
            result.append(" ".join(merged_parts).strip())
        else:
            result.append(chunks[i])
        i = j

    return result


def _can_group_elements(left: ParsedElement, right: ParsedElement) -> bool:
    if left.section_path != right.section_path:
        return False
    return True


def _group_page_number(elements: list[ParsedElement]) -> int | None:
    page_numbers = {element.page_number for element in elements if element.page_number is not None}
    if len(page_numbers) == 1:
        return next(iter(page_numbers))
    return None


def _element_group_metadata(elements: list[ParsedElement]) -> dict[str, object]:
    metadata: dict[str, object] = {}

    start_offsets = [element.start_char for element in elements if element.start_char is not None]
    end_offsets = [element.end_char for element in elements if element.end_char is not None]
    if start_offsets:
        metadata["start_char"] = min(start_offsets)
    if end_offsets:
        metadata["end_char"] = max(end_offsets)

    if len(elements) == 1:
        metadata.update(elements[0].metadata)

    return metadata
