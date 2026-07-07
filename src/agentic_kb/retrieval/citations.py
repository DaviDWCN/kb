"""Build source citations from selected retrieval hits."""

from dataclasses import dataclass

from agentic_kb.schemas.search import SearchHit


@dataclass(frozen=True)
class CitationSource:
    """Rich source reference suitable for answer citations and API responses."""

    document_id: str
    chunk_id: str
    score: float
    source_uri: str | None = None
    heading_path: tuple[str, ...] = ()
    page_number: int | None = None
    chunk_index: int | None = None


class CitationBuilder:
    """Build rich citation sources from selected context hits."""

    def __init__(self, *, dedupe_by_chunk: bool = True) -> None:
        self._dedupe_by_chunk = dedupe_by_chunk

    def build(self, hits: list[SearchHit]) -> list[CitationSource]:
        sources: list[CitationSource] = []
        seen_chunk_ids: set[str] = set()

        for hit in hits:
            if self._dedupe_by_chunk and hit.chunk.id in seen_chunk_ids:
                continue

            seen_chunk_ids.add(hit.chunk.id)
            sources.append(_source_from_hit(hit))

        return sources


def _source_from_hit(hit: SearchHit) -> CitationSource:
    chunk = hit.chunk
    source_uri = chunk.metadata.attributes.get("source_uri")
    if not isinstance(source_uri, str):
        source_uri = None

    return CitationSource(
        document_id=chunk.document_id,
        chunk_id=chunk.id,
        score=hit.citation.score if hit.citation is not None else hit.score,
        source_uri=source_uri,
        heading_path=chunk.metadata.heading_path,
        page_number=chunk.metadata.page_number,
        chunk_index=chunk.index,
    )
