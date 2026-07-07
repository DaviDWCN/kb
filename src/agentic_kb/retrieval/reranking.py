"""Cross-encoder reranking for retrieved candidate chunks."""

import os

from agentic_kb.schemas.search import Citation, SearchHit


class CrossEncoderReranker:
    """Rerank candidate hits with a cross-encoder pair scoring model."""

    def __init__(self, model) -> None:
        self._model = model

    @classmethod
    def from_sentence_transformers(cls, model_name: str) -> "CrossEncoderReranker":
        """Build a reranker from sentence-transformers when the optional extra exists."""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker; "
                "install agentic-kb[reranking]."
            ) from error

        return cls(CrossEncoder(model_name))

    def rerank(
        self,
        query: str,
        hits: list[SearchHit],
        *,
        k: int | None = None,
    ) -> list[SearchHit]:
        """Score query/chunk text pairs and return hits sorted by relevance."""
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("query must not be empty")
        if not hits or (k is not None and k <= 0):
            return []

        pairs = [(cleaned_query, hit.chunk.text) for hit in hits]
        scores = [float(score) for score in self._model.predict(pairs)]
        if len(scores) != len(pairs):
            raise ValueError(
                f"cross-encoder returned {len(scores)} scores for {len(pairs)} pairs"
            )

        return _rerank_hits(hits, scores, k=k)


class ApiReranker:
    """Rerank via an OpenAI-compatible /v1/rerank endpoint (e.g. GPUSTack)."""

    def __init__(
        self,
        model: str,
        *,
        api_key_env: str = "RERANKER_API_KEY",
        base_url_env: str = "RERANKER_BASE_URL",
        timeout: float = 30.0,
    ) -> None:
        self._model = model
        self._url = f"{(os.getenv(base_url_env, '')).rstrip('/')}/rerank"
        self._headers = {"Authorization": f"Bearer {os.getenv(api_key_env, '')}"}
        self._timeout = timeout

    def rerank(self, query: str, hits: list[SearchHit], *, k: int | None = None) -> list[SearchHit]:
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("query must not be empty")
        if not hits or (k is not None and k <= 0):
            return []

        import requests as _requests  # noqa: F811 — optional dependency

        documents = [hit.chunk.text for hit in hits]
        resp = _requests.post(
            self._url,
            headers=self._headers,
            json={"model": self._model, "query": cleaned_query, "documents": documents},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        scores = [_extract_rerank_score(results, i) for i in range(len(documents))]
        return _rerank_hits(hits, scores, k=k)


def _extract_rerank_score(results: list[dict], index: int) -> float:
    for result in results:
        if result.get("index") == index:
            return float(result.get("relevance_score", 0.0))
    return 0.0


def _rerank_hits(
    hits: list[SearchHit],
    scores: list[float],
    *,
    k: int | None,
) -> list[SearchHit]:
    reranked = [
        SearchHit(
            chunk=hit.chunk,
            score=score,
            citation=_citation_with_score(hit, score),
            highlights=hit.highlights,
        )
        for hit, score in zip(hits, scores)
    ]
    reranked.sort(key=lambda hit: (-hit.score, hit.chunk.index, hit.chunk.id))
    return reranked if k is None else reranked[:k]


def _citation_with_score(hit: SearchHit, score: float) -> Citation:
    if hit.citation is None:
        return Citation(document_id=hit.chunk.document_id, chunk_id=hit.chunk.id, score=score)

    return Citation(
        document_id=hit.citation.document_id,
        chunk_id=hit.citation.chunk_id,
        score=score,
        span=hit.citation.span,
    )
