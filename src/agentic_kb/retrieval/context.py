"""Context selection for answer-generation prompts."""

from dataclasses import dataclass

from agentic_kb.schemas.search import SearchHit


@dataclass(frozen=True)
class ContextSelection:
    """Selected context hits and budget accounting."""

    selected: list[SearchHit]
    omitted: list[SearchHit]
    token_count: int
    max_tokens: int


class ContextSelector:
    """Choose which reranked hits fit into a context token budget."""

    def __init__(
        self,
        *,
        max_tokens: int,
        reserve_tokens: int = 0,
        dedupe_by_chunk: bool = True,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        if reserve_tokens < 0:
            raise ValueError("reserve_tokens must not be negative")
        if reserve_tokens >= max_tokens:
            raise ValueError("reserve_tokens must be smaller than max_tokens")

        self._max_tokens = max_tokens
        self._usable_tokens = max_tokens - reserve_tokens
        self._dedupe_by_chunk = dedupe_by_chunk

    def select(self, hits: list[SearchHit]) -> ContextSelection:
        """Select the highest-ranked hits that fit within the usable token budget."""
        selected: list[SearchHit] = []
        omitted: list[SearchHit] = []
        seen_chunk_ids: set[str] = set()
        token_count = 0

        for hit in hits:
            if self._dedupe_by_chunk and hit.chunk.id in seen_chunk_ids:
                omitted.append(hit)
                continue

            hit_token_count = _token_count(hit)
            if token_count + hit_token_count > self._usable_tokens:
                omitted.append(hit)
                continue

            selected.append(hit)
            seen_chunk_ids.add(hit.chunk.id)
            token_count += hit_token_count

        return ContextSelection(
            selected=selected,
            omitted=omitted,
            token_count=token_count,
            max_tokens=self._usable_tokens,
        )


def _token_count(hit: SearchHit) -> int:
    if hit.chunk.metadata.token_count is not None:
        return hit.chunk.metadata.token_count
    return len(hit.chunk.text.split())
