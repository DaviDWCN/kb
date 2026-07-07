"""Dense vector retrieval over embedded queries."""

from agentic_kb.embeddings import EmbeddingService
from agentic_kb.schemas.vectors import Metadata, SearchResult
from agentic_kb.vector_store import VectorStore


class DenseRetriever:
    """Search a vector store by embedding incoming query text."""

    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        filter: Metadata | None = None,
    ) -> list[SearchResult]:
        """Embed a query and return ranked vector-store matches."""
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("query must not be empty")
        if k <= 0:
            return []

        query_embedding = self._embedding_service.embed_texts([cleaned_query])[0]
        return self._vector_store.search(query_embedding, k, filter=filter)
