"""FAISS-backed in-memory vector store."""

import importlib
from typing import Any

from agentic_kb.schemas.vectors import Embedding, Metadata, SearchResult, VectorRecord
from agentic_kb.vector_store.base import VectorStore


class FaissVectorStore(VectorStore):
    """Vector store backed by FAISS ``IndexFlatIP`` with cosine scoring.

    FAISS stores only numeric vectors, so this class keeps the authoritative
    ``VectorRecord`` objects in Python dictionaries and uses FAISS for fast
    nearest-neighbor scoring. Vectors are normalized before insertion and query
    so FAISS inner product scores match cosine similarity.
    """

    def __init__(self, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")

        self._dimensions = dimensions
        self._faiss = _load_faiss()
        self._np = _load_numpy()
        self._records: dict[str, VectorRecord] = {}
        self._chunk_ids: list[str] = []
        self._index = self._new_index()

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def upsert(self, record: VectorRecord) -> None:
        self._validate_embedding(record.embedding)
        self._records[record.chunk_id] = record
        self._rebuild_index()

    def delete(self, chunk_id: str) -> None:
        if chunk_id in self._records:
            self._records.pop(chunk_id)
            self._rebuild_index()

    def update(
        self,
        chunk_id: str,
        *,
        embedding: Embedding | None = None,
        metadata: Metadata | None = None,
    ) -> None:
        current = self._records[chunk_id]
        next_embedding = current.embedding if embedding is None else embedding
        next_metadata = current.metadata if metadata is None else metadata
        self._validate_embedding(next_embedding)
        self._records[chunk_id] = VectorRecord(
            chunk_id=current.chunk_id,
            document_id=current.document_id,
            embedding=next_embedding,
            metadata=next_metadata,
        )
        self._rebuild_index()

    def search(
        self,
        query: Embedding,
        k: int,
        *,
        filter: Metadata | None = None,
    ) -> list[SearchResult]:
        if k <= 0:
            return []

        self._validate_embedding(query)
        if not self._records:
            return []

        # Metadata filtering happens outside FAISS. Search the full in-memory
        # index first, then filter, so lower-ranked matching records are not
        # accidentally hidden behind higher-ranked non-matching records.
        search_limit = len(self._chunk_ids)
        query_vector = self._as_matrix([query])
        distances, indices = self._index.search(query_vector, search_limit)

        results: list[SearchResult] = []
        for score, index in zip(distances[0], indices[0]):
            if index < 0:
                continue

            chunk_id = self._chunk_ids[int(index)]
            record = self._records[chunk_id]
            if not _matches_filter(record.metadata, filter):
                continue

            results.append(SearchResult(record=record, score=float(score)))
            if len(results) == k:
                break

        return results

    def get(self, chunk_id: str) -> VectorRecord | None:
        return self._records.get(chunk_id)

    def __len__(self) -> int:
        return len(self._records)

    def _rebuild_index(self) -> None:
        self._index = self._new_index()
        self._chunk_ids = list(self._records.keys())
        if not self._chunk_ids:
            return

        matrix = self._as_matrix([self._records[chunk_id].embedding for chunk_id in self._chunk_ids])
        self._index.add(matrix)

    def _new_index(self) -> Any:
        return self._faiss.IndexFlatIP(self._dimensions)

    def _as_matrix(self, embeddings: list[Embedding]) -> Any:
        matrix = self._np.asarray(embeddings, dtype=self._np.float32)
        return _normalize_rows(matrix, self._np)

    def _validate_embedding(self, embedding: Embedding) -> None:
        if len(embedding) != self._dimensions:
            raise ValueError(
                f"embedding has {len(embedding)} dimensions, expected {self._dimensions}"
            )


def _load_faiss() -> Any:
    try:
        return importlib.import_module("faiss")
    except ImportError as error:
        raise ImportError(
            "faiss-cpu is required for FaissVectorStore; install agentic-kb[vector]."
        ) from error


def _load_numpy() -> Any:
    try:
        return importlib.import_module("numpy")
    except ImportError as error:
        raise ImportError("numpy is required for FaissVectorStore.") from error


def _normalize_rows(matrix: Any, np: Any) -> Any:
    norms = np.sqrt((matrix * matrix).sum(axis=1, keepdims=True))
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)


def _matches_filter(metadata: Metadata, filter: Metadata | None) -> bool:
    if filter is None:
        return True
    return all(metadata.get(key) == value for key, value in filter.items())
