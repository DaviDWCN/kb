"""Pipeline orchestration for the in-memory RAG flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from agentic_kb.embeddings import EmbeddingIndexer, EmbeddingService
from agentic_kb.retrieval.chunk_store import InMemoryChunkStore
from agentic_kb.retrieval.context import ContextSelection, ContextSelector
from agentic_kb.retrieval.dense import DenseRetriever
from agentic_kb.retrieval.hybrid import HybridRetriever
from agentic_kb.retrieval.sparse import SparseRetriever
from agentic_kb.schemas.chunks import Chunk
from agentic_kb.schemas.search import Citation, SearchFilters, SearchHit
from agentic_kb.schemas.vectors import VectorRecord
from agentic_kb.vector_store import VectorStore

if TYPE_CHECKING:
    from agentic_kb.generation.answering import Answer


class AnswerGeneratorProtocol(Protocol):
    """Answer-generation dependency expected by the RAG pipeline."""

    def answer(self, query: str, context: ContextSelection) -> "Answer":
        ...


class RerankerProtocol(Protocol):
    """Reranking dependency expected by the RAG pipeline."""

    def rerank(self, query: str, hits: list[SearchHit], *, k: int | None = None) -> list[SearchHit]:
        ...


@dataclass(frozen=True)
class RagPipelineResult:
    """Full trace of one RAG query through retrieval, context, and answer generation."""

    answer: "Answer"
    candidates: list[SearchHit]
    reranked: list[SearchHit]
    context: ContextSelection


class RagPipeline:
    """Coordinate indexing, retrieval, optional reranking, and answer generation.

    The pipeline is intentionally orchestration-only: embeddings, vector search,
    sparse search, reranking, context budgeting, and prompting remain owned by
    their focused components. This keeps the end-to-end flow easy to test while
    still allowing each provider-backed component to be swapped independently.
    """

    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        answer_generator: AnswerGeneratorProtocol,
        context_selector: ContextSelector,
        reranker: RerankerProtocol | None = None,
        chunk_store: InMemoryChunkStore | None = None,
        search_k: int = 5,
        dense_only: bool = False,
        search_width: int | None = None,
    ) -> None:
        if search_k <= 0:
            raise ValueError("search_k must be greater than zero")

        self._embedding_indexer = EmbeddingIndexer(embedding_service, vector_store)
        self._dense_retriever = DenseRetriever(embedding_service, vector_store)
        self._answer_generator = answer_generator
        self._context_selector = context_selector
        self._reranker = reranker
        self._search_k = search_k
        self._search_width = search_width if search_width is not None else search_k
        self._chunk_store = chunk_store if chunk_store is not None else InMemoryChunkStore()
        self._sparse_retriever: SparseRetriever | None = None  # lazily built, cached
        self._dense_only = dense_only

    def index_chunks(self, chunks: list[Chunk]) -> list[VectorRecord]:
        """Embed chunks, write vectors, and update the in-memory chunk index."""
        records = self._embedding_indexer.refresh_chunks(chunks)
        self._chunk_store.upsert_many(chunks)
        self._sparse_retriever = None  # invalidate cached BM25 index
        return records

    def answer(
        self,
        query: str,
        *,
        k: int | None = None,
        filter: SearchFilters | None = None,
        expanded_query: list[str] | str | None = None,
        sparse_query: str | None = None,
    ) -> RagPipelineResult:
        """Run one query through retrieval, context selection, and answer generation.

        When *expanded_query* is a list of variant queries, each is retrieved
        independently and results are merged with deduplication.
        *sparse_query* uses LLM-extracted keywords for BM25 retrieval.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("query must not be empty")

        limit = self._resolve_limit(k)
        search_limit = self._search_width
        if isinstance(expanded_query, list) and expanded_query:
            candidates = self._multi_retrieve(
                cleaned_query, expanded_query, k=search_limit, filter=filter, sparse_query=sparse_query,
            )
        else:
            search_query = expanded_query.strip() if isinstance(expanded_query, str) and expanded_query else cleaned_query
            candidates = self._retrieve(search_query, k=search_limit, filter=filter, sparse_query=sparse_query)

        # Multi-hop: ask LLM what's missing and retrieve again.
        preliminary = self._context_selector.select(candidates)
        gap = self._gap_query(cleaned_query, preliminary)
        if gap:
            gap_candidates = self._retrieve(gap, k=search_limit, filter=filter, sparse_query=sparse_query)
            seen: dict[str, SearchHit] = {h.chunk.id: h for h in candidates}
            for h in gap_candidates:
                if h.chunk.id not in seen or h.score > seen[h.chunk.id].score:
                    seen[h.chunk.id] = h
            candidates = sorted(seen.values(), key=lambda h: (-h.score, h.chunk.index, h.chunk.id))

        reranked = self._rerank(cleaned_query, candidates, k=limit)
        # reranked = _dedup_near_duplicates(reranked)
        context = self._context_selector.select(reranked)
        answer = self._answer_generator.answer(cleaned_query, context)

        return RagPipelineResult(
            answer=answer,
            candidates=candidates,
            reranked=reranked,
            context=context,
        )

    def _gap_query(self, query: str, context: ContextSelection) -> str | None:
        """Ask the LLM what information is still missing to answer the question."""
        if not context.selected:
            return None
        try:
            chunk_preview = "\n".join(
                h.chunk.text[:200] for h in context.selected[:3]
            )
            prompt = (
                "根据以下问题和已检索到的文档片段，判断是否缺少关键信息。"
                "如果缺少，生成一个简短的搜索查询来查找缺失信息（不超过30字）。"
                "如果不缺，回复 NONE。\n\n"
                f"问题: {query}\n\n已检索片段:\n{chunk_preview}"
            )
            response = self._answer_generator._model.generate(prompt).strip()
        except Exception:
            return None
        if not response or response.upper() == "NONE" or len(response) > 100:
            return None
        return response

    def _multi_retrieve(
        self,
        query: str,
        variants: list[str],
        *,
        k: int,
        filter: SearchFilters | None,
        sparse_query: str | None = None,
    ) -> list[SearchHit]:
        """Retrieve for each variant and merge by best-score deduplication."""
        seen: dict[str, SearchHit] = {}
        for variant in variants:
            for hit in self._retrieve(variant, k=k, filter=filter, sparse_query=sparse_query):
                if hit.chunk.id not in seen or hit.score > seen[hit.chunk.id].score:
                    seen[hit.chunk.id] = hit
        merged = sorted(seen.values(), key=lambda h: (-h.score, h.chunk.index, h.chunk.id))
        return merged[:k]

    def _retrieve(
        self,
        query: str,
        *,
        k: int,
        filter: SearchFilters | None,
        sparse_query: str | None = None,
    ) -> list[SearchHit]:
        if self._dense_only:
            return self._dense_retrieve(query, k=k, filter=filter)

        # Build the sparse retriever lazily and cache it across queries.
        # Invalidated only when index_chunks adds new data.
        if self._sparse_retriever is None:
            self._sparse_retriever = SparseRetriever(self._chunk_store.ordered())

        hybrid_retriever = HybridRetriever(
            self._dense_retriever,
            self._sparse_retriever,
            chunks_by_id=self._chunk_store.as_dict(),
        )
        return hybrid_retriever.search(query, k=k, filter=filter, sparse_query=sparse_query)

    def _dense_retrieve(
        self,
        query: str,
        *,
        k: int,
        filter: SearchFilters | None,
    ) -> list[SearchHit]:
        """Dense-only retrieval: embed query, FAISS search, hydrate chunks."""
        metadata_filter = None
        if filter:
            metadata_filter = dict(filter)

        results = self._dense_retriever.search(query, k=k, filter=metadata_filter)
        hits: list[SearchHit] = []
        for result in results:
            chunk = self._chunk_store.get(result.record.chunk_id)
            if chunk is None:
                continue
            hits.append(
                SearchHit(
                    chunk=chunk,
                    score=result.score,
                    citation=Citation(
                        document_id=chunk.document_id,
                        chunk_id=chunk.id,
                        score=result.score,
                    ),
                    highlights=(),
                )
            )
        return hits

    def _rerank(self, query: str, hits: list[SearchHit], *, k: int) -> list[SearchHit]:
        if self._reranker is None:
            return hits[:k]
        return self._reranker.rerank(query, hits, k=k)

    def _resolve_limit(self, k: int | None) -> int:
        if k is None:
            return self._search_k
        if k <= 0:
            return 0
        return k


def _dedup_near_duplicates(hits: list[SearchHit], threshold: float = 0.85) -> list[SearchHit]:
    """Remove chunks whose text is near-identical to a higher-ranked chunk.

    Uses character trigram Jaccard similarity on the first 200 chars of each
    chunk.  The first occurrence (highest score) is kept.
    """
    if len(hits) <= 1:
        return hits

    kept: list[SearchHit] = [hits[0]]
    kept_sigs: list[set[str]] = [_trigrams(hits[0].chunk.text[:200])]

    for hit in hits[1:]:
        sig = _trigrams(hit.chunk.text[:200])
        if not sig:
            kept.append(hit)
            kept_sigs.append(sig)
            continue
        if all(_jaccard(sig, ks) > threshold for ks in kept_sigs):
            kept.append(hit)
            kept_sigs.append(sig)

    return kept


def _trigrams(text: str) -> set[str]:
    """Character trigrams for Chinese text similarity."""
    t = text.replace("\n", " ").replace(" ", "")
    return {t[i:i+3] for i in range(len(t) - 2)} if len(t) >= 3 else {t}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 1.0
    return len(a & b) / len(a | b)
