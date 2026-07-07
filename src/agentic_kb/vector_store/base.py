"""Vector store abstraction used by retrieval and indexing."""

from abc import ABC, abstractmethod

from agentic_kb.schemas.vectors import Embedding, Metadata, SearchResult, VectorRecord


class VectorStore(ABC):
    """Interface every vector index implementation must provide."""

    @abstractmethod
    def upsert(self, record: VectorRecord) -> None:
        """Insert or replace a vector record by chunk ID."""

        raise NotImplementedError

    def upsert_many(self, records: list[VectorRecord]) -> None:
        """Insert or replace multiple vector records. Default loops over upsert()."""

        for record in records:
            self.upsert(record)

    @abstractmethod
    def delete(self, chunk_id: str) -> None:
        """Remove a vector record if it exists."""

        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        chunk_id: str,
        *,
        embedding: Embedding | None = None,
        metadata: Metadata | None = None,
    ) -> None:
        """Update embedding and/or metadata for an existing vector record."""

        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query: Embedding,
        k: int,
        *,
        filter: Metadata | None = None,
    ) -> list[SearchResult]:
        """Return the most similar records for an embedding query."""

        raise NotImplementedError
