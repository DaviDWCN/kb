"""Sparse keyword retrieval over chunk text."""

import math
import re
import warnings
from collections import Counter

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
    import jieba

from agentic_kb.schemas.chunks import Chunk
from agentic_kb.schemas.search import Citation, SearchFilters, SearchHit


_BM25_K1 = 1.5
_BM25_B = 0.75


class SparseRetriever:
    """Search chunks with lexical keyword matching."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self._term_counts = [_term_counts(chunk.text) for chunk in self._chunks]
        self._document_frequencies = _document_frequencies(self._term_counts)
        self._document_lengths = [sum(term_counts.values()) for term_counts in self._term_counts]
        self._average_document_length = _average(self._document_lengths)

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        filter: SearchFilters | None = None,
    ) -> list[SearchHit]:
        """Return chunks ranked by BM25-style keyword relevance."""
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("query must not be empty")
        if k <= 0:
            return []

        query_terms = tuple(dict.fromkeys(_tokenize(cleaned_query)))
        if not query_terms:
            raise ValueError("query must not be empty")

        hits: list[SearchHit] = []
        for chunk, term_counts, document_length in zip(
            self._chunks,
            self._term_counts,
            self._document_lengths,
        ):
            if not _matches_filter(chunk, filter):
                continue

            score = self._score(query_terms, term_counts, document_length)
            if score <= 0:
                continue

            hits.append(
                SearchHit(
                    chunk=chunk,
                    score=score,
                    citation=Citation(
                        document_id=chunk.document_id,
                        chunk_id=chunk.id,
                        score=score,
                    ),
                    highlights=tuple(term for term in query_terms if term in term_counts),
                )
            )

        hits.sort(key=lambda hit: (-hit.score, hit.chunk.index, hit.chunk.id))
        return hits[:k]

    def _score(
        self,
        query_terms: tuple[str, ...],
        term_counts: Counter[str],
        document_length: int,
    ) -> float:
        if self._average_document_length == 0:
            return 0.0

        score = 0.0
        for term in query_terms:
            frequency = term_counts[term]
            if frequency == 0:
                continue

            idf = _inverse_document_frequency(
                total_documents=len(self._chunks),
                document_frequency=self._document_frequencies[term],
            )
            length_normalizer = 1 - _BM25_B + _BM25_B * (
                document_length / self._average_document_length
            )
            score += idf * ((frequency * (_BM25_K1 + 1)) / (frequency + _BM25_K1 * length_normalizer))

        return score


def _tokenize(text: str) -> list[str]:
    """Tokenize with jieba for CJK word segmentation and regex for ASCII."""
    tokens = jieba.lcut(text.lower())
    # Keep only tokens containing alphanumeric or CJK content; drop whitespace/punctuation-only.
    return [t for t in tokens if t.strip() and re.search(r"[a-z0-9\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]", t)]


def _term_counts(text: str) -> Counter[str]:
    return Counter(_tokenize(text))


def _document_frequencies(term_counts_by_chunk: list[Counter[str]]) -> Counter[str]:
    frequencies: Counter[str] = Counter()
    for term_counts in term_counts_by_chunk:
        frequencies.update(term_counts.keys())
    return frequencies


def _inverse_document_frequency(total_documents: int, document_frequency: int) -> float:
    return math.log(1 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5))


def _average(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _matches_filter(chunk: Chunk, filter: SearchFilters | None) -> bool:
    if filter is None:
        return True

    metadata = _filterable_metadata(chunk)
    return all(metadata.get(key) == value for key, value in filter.items())


def _filterable_metadata(chunk: Chunk) -> dict[str, object]:
    metadata: dict[str, object] = dict(chunk.metadata.attributes)
    metadata.update(
        {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "chunk_index": chunk.index,
            "heading_path": chunk.metadata.heading_path,
            "page_number": chunk.metadata.page_number,
            "token_count": chunk.metadata.token_count,
        }
    )
    return metadata
