"""Index chunk embeddings into a vector store."""

from agentic_kb.chunking.contextual import text_for_indexing
from agentic_kb.embeddings.service import EmbeddingService
from agentic_kb.schemas.chunks import Chunk, ChunkMetadata
from agentic_kb.schemas.vectors import Metadata, VectorRecord
from agentic_kb.vector_store import VectorStore


_VECTOR_METADATA_ATTRIBUTE_KEYS = {
    "content_type",
    "contextual_summary",
    "effective_date",
    "expiry_date",
    "file_mtime",
    "metadata_confidence",
    "metadata_source",
    "section",
    "source_uri",
}


class EmbeddingIndexer:
    """Embed chunks and upsert their vectors into the configured store."""

    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    def refresh_chunks(self, chunks: list[Chunk]) -> list[VectorRecord]:
        """Embed chunks, batch-upsert their vector records, and return stored records."""
        if not chunks:
            return []

        embeddings = self._embedding_service.embed_texts([text_for_indexing(chunk) for chunk in chunks])

        records: list[VectorRecord] = []
        for chunk, embedding in zip(chunks, embeddings):
            record = VectorRecord(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                embedding=embedding,
                metadata=_metadata_from_chunk(chunk),
            )
            records.append(record)

        self._vector_store.upsert_many(records)
        return records


def _metadata_from_chunk(chunk: Chunk) -> Metadata:
    """Flatten chunk metadata into vector-store metadata for filtering/hydration."""
    metadata = _metadata_fields(chunk.metadata)
    metadata["chunk_index"] = chunk.index
    return metadata


def _metadata_fields(chunk_metadata: ChunkMetadata) -> Metadata:
    metadata: Metadata = {
        key: chunk_metadata.attributes[key]
        for key in _VECTOR_METADATA_ATTRIBUTE_KEYS
        if key in chunk_metadata.attributes
    }
    metadata["heading_path"] = chunk_metadata.heading_path
    if chunk_metadata.page_number is not None:
        metadata["page_number"] = chunk_metadata.page_number
    if chunk_metadata.token_count is not None:
        metadata["token_count"] = chunk_metadata.token_count
    return metadata
