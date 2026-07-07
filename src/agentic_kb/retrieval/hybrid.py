"""Hybrid retrieval by fusing dense and sparse rankings."""

from agentic_kb.retrieval.dense import DenseRetriever
from agentic_kb.retrieval.sparse import SparseRetriever
from agentic_kb.schemas.chunks import Chunk
from agentic_kb.schemas.search import Citation, SearchFilters, SearchHit


class HybridRetriever:
    """Merge dense and sparse retriever results into one ranked list."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        *,
        chunks_by_id: dict[str, Chunk],
        rrf_k: int = 60,
    ) -> None:
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")

        self._dense_retriever = dense_retriever
        self._sparse_retriever = sparse_retriever
        self._chunks_by_id = dict(chunks_by_id)
        self._rrf_k = rrf_k

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        filter: SearchFilters | None = None,
        sparse_query: str | None = None,
    ) -> list[SearchHit]:
        """Return dense/sparse results fused with reciprocal rank fusion.

        *sparse_query* uses keyword-style text for BM25 when provided;
        *query* is always used for dense retrieval.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("query must not be empty")
        if k <= 0:
            return []

        sparse_text = (sparse_query or "").strip() or cleaned_query
        dense_results = self._dense_retriever.search(cleaned_query, k=k)
        sparse_hits = self._sparse_retriever.search(sparse_text, k=k, filter=filter)

        chunks: dict[str, Chunk] = {}
        scores: dict[str, float] = {}
        highlights: dict[str, tuple[str, ...]] = {}

        for rank, result in enumerate(dense_results, start=1):
            chunk = self._chunks_by_id.get(result.record.chunk_id)
            if chunk is None:
                continue
            if not _filter_match(chunk, filter):
                continue
            chunks[chunk.id] = chunk
            scores[chunk.id] = scores.get(chunk.id, 0.0) + self._rank_score(rank)

        for rank, hit in enumerate(sparse_hits, start=1):
            chunks[hit.chunk.id] = hit.chunk
            scores[hit.chunk.id] = scores.get(hit.chunk.id, 0.0) + self._rank_score(rank)
            if hit.highlights:
                highlights[hit.chunk.id] = hit.highlights

        fused_hits = [
            SearchHit(
                chunk=chunk,
                score=scores[chunk_id],
                citation=Citation(
                    document_id=chunk.document_id,
                    chunk_id=chunk.id,
                    score=scores[chunk_id],
                ),
                highlights=highlights.get(chunk_id, ()),
            )
            for chunk_id, chunk in chunks.items()
        ]
        fused_hits.sort(key=lambda hit: (-hit.score, hit.chunk.index, hit.chunk.id))
        return fused_hits[:k]

    def _rank_score(self, rank: int) -> float:
        return 1.0 / (self._rrf_k + rank)


def _filter_match(chunk: Chunk, filter: SearchFilters | None) -> bool:
    if filter is None:
        return True
    metadata: dict[str, object] = dict(chunk.metadata.attributes)
    metadata.update({
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "chunk_index": chunk.index,
        "heading_path": chunk.metadata.heading_path,
        "page_number": chunk.metadata.page_number,
        "token_count": chunk.metadata.token_count,
    })
    return all(metadata.get(key) == value for key, value in filter.items())
