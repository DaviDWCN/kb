"""Ranked retrieval evaluation metrics for RAG pipelines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_kb.schemas.search import SearchHit


@dataclass(frozen=True)
class RetrievalCase:
    """Expected relevant chunks and actual ranked retrieval output for one query."""

    id: str
    query: str
    relevant_chunk_ids: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]

    @classmethod
    def from_hits(
        cls,
        *,
        id: str,
        query: str,
        relevant_chunk_ids: tuple[str, ...],
        hits: list["SearchHit"],
    ) -> "RetrievalCase":
        """Build a retrieval case from pipeline SearchHit objects."""
        return cls(
            id=id,
            query=query,
            relevant_chunk_ids=relevant_chunk_ids,
            retrieved_chunk_ids=tuple(hit.chunk.id for hit in hits),
        )


@dataclass(frozen=True)
class RetrievalResult:
    """Retrieval metrics for one case at a fixed cutoff."""

    case_id: str
    k: int
    hit_at_k: bool
    precision_at_k: float
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    relevant_retrieved_count: int
    relevant_count: int
    retrieved_count: int


@dataclass(frozen=True)
class RetrievalSummary:
    """Aggregate retrieval metrics across cases."""

    results: list[RetrievalResult]
    k: int
    case_count: int
    hit_rate_at_k: float
    average_precision_at_k: float
    average_recall_at_k: float
    average_mrr_at_k: float
    average_ndcg_at_k: float


def evaluate_retrieval(cases: list[RetrievalCase], *, k: int) -> RetrievalSummary:
    """Evaluate ranked retrieval results with standard binary relevance metrics."""
    if k <= 0:
        raise ValueError("k must be greater than zero")

    results = [_evaluate_case(case, k=k) for case in cases]
    if not results:
        return RetrievalSummary(
            results=[],
            k=k,
            case_count=0,
            hit_rate_at_k=0.0,
            average_precision_at_k=0.0,
            average_recall_at_k=0.0,
            average_mrr_at_k=0.0,
            average_ndcg_at_k=0.0,
        )

    return RetrievalSummary(
        results=results,
        k=k,
        case_count=len(results),
        hit_rate_at_k=sum(1 for result in results if result.hit_at_k) / len(results),
        average_precision_at_k=_average(result.precision_at_k for result in results),
        average_recall_at_k=_average(result.recall_at_k for result in results),
        average_mrr_at_k=_average(result.mrr_at_k for result in results),
        average_ndcg_at_k=_average(result.ndcg_at_k for result in results),
    )


def _evaluate_case(case: RetrievalCase, *, k: int) -> RetrievalResult:
    _validate_case(case)
    relevant_ids = set(case.relevant_chunk_ids)
    top_ids = case.retrieved_chunk_ids[:k]
    relevance = _binary_relevance(top_ids, relevant_ids)
    relevant_retrieved_count = sum(1 for value in relevance if value)
    relevant_count = len(relevant_ids)

    return RetrievalResult(
        case_id=case.id,
        k=k,
        hit_at_k=relevant_retrieved_count > 0,
        precision_at_k=relevant_retrieved_count / k,
        recall_at_k=relevant_retrieved_count / relevant_count,
        mrr_at_k=_mrr(relevance),
        ndcg_at_k=_ndcg(relevance, ideal_relevant_count=min(relevant_count, k)),
        relevant_retrieved_count=relevant_retrieved_count,
        relevant_count=relevant_count,
        retrieved_count=len(top_ids),
    )


def _binary_relevance(retrieved_ids: tuple[str, ...], relevant_ids: set[str]) -> list[int]:
    """Mark first occurrence of each relevant retrieved ID; duplicates do not score twice."""
    seen_relevant: set[str] = set()
    relevance: list[int] = []
    for chunk_id in retrieved_ids:
        if chunk_id in relevant_ids and chunk_id not in seen_relevant:
            relevance.append(1)
            seen_relevant.add(chunk_id)
            continue
        relevance.append(0)
    return relevance


def _mrr(relevance: list[int]) -> float:
    for index, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            return 1 / index
    return 0.0


def _ndcg(relevance: list[int], *, ideal_relevant_count: int) -> float:
    if ideal_relevant_count <= 0:
        return 0.0
    dcg = sum(value / math.log2(rank + 1) for rank, value in enumerate(relevance, start=1))
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_relevant_count + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def _validate_case(case: RetrievalCase) -> None:
    if not case.id.strip():
        raise ValueError("case id must not be empty")
    if not case.query.strip():
        raise ValueError("query must not be empty")
    if not case.relevant_chunk_ids:
        raise ValueError("relevant_chunk_ids must not be empty")


def _average(values) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0
