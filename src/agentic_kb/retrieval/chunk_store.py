"""In-memory chunk lookup used to hydrate retrieval results."""

import json
import sqlite3
from pathlib import Path

from agentic_kb.schemas.chunks import Chunk
from agentic_kb.schemas.chunks import ChunkMetadata


class InMemoryChunkStore:
    """Keep the current chunk set addressable by chunk ID.

    Dense vector stores normally return IDs and scores, not full chunk text.
    This small map is the in-memory hydration layer that turns those IDs back
    into canonical ``Chunk`` objects for sparse search, reranking, context
    selection, citations, and answer generation.
    """

    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self._chunks_by_id: dict[str, Chunk] = {}
        if chunks:
            self.upsert_many(chunks)

    def upsert_many(self, chunks: list[Chunk]) -> None:
        """Insert or replace chunks by their stable chunk IDs."""
        for chunk in chunks:
            self._chunks_by_id[chunk.id] = chunk

    def get(self, chunk_id: str) -> Chunk | None:
        """Return a chunk by ID, or ``None`` when the chunk is not indexed."""
        return self._chunks_by_id.get(chunk_id)

    def as_dict(self) -> dict[str, Chunk]:
        """Return a defensive copy for components that expect ``chunks_by_id``."""
        return dict(self._chunks_by_id)

    def ordered(self) -> list[Chunk]:
        """Return chunks in stable source order for deterministic sparse indexing."""
        return sorted(
            self._chunks_by_id.values(),
            key=lambda chunk: (chunk.document_id, chunk.index, chunk.id),
        )

    def __len__(self) -> int:
        return len(self._chunks_by_id)


class SQLiteChunkStore:
    """Persist chunks locally so FAISS search results can be hydrated after restart."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self._db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._ensure_schema()

    def upsert_many(self, chunks: list[Chunk]) -> None:
        """Insert or replace chunks by ID."""
        with self._connection:
            self._connection.executemany(
                """
                INSERT OR REPLACE INTO chunks (
                    chunk_id, document_id, text, chunk_index, metadata_json
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.text,
                        chunk.index,
                        _metadata_json(chunk.metadata),
                    )
                    for chunk in chunks
                ],
            )

    def get(self, chunk_id: str) -> Chunk | None:
        row = self._connection.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return None
        return _chunk_from_row(row)

    def as_dict(self) -> dict[str, Chunk]:
        return {chunk.id: chunk for chunk in self.ordered()}

    def ordered(self) -> list[Chunk]:
        rows = self._connection.execute(
            """
            SELECT * FROM chunks
            ORDER BY document_id, chunk_index, chunk_id
            """
        ).fetchall()
        return [_chunk_from_row(row) for row in rows]

    def __len__(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
        return int(row["count"])

    def _ensure_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )


def _metadata_json(metadata: ChunkMetadata) -> str:
    return json.dumps(
        {
            "heading_path": list(metadata.heading_path),
            "page_number": metadata.page_number,
            "token_count": metadata.token_count,
            "attributes": metadata.attributes,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _chunk_from_row(row: sqlite3.Row) -> Chunk:
    metadata = json.loads(row["metadata_json"])
    return Chunk(
        id=row["chunk_id"],
        document_id=row["document_id"],
        text=row["text"],
        index=row["chunk_index"],
        metadata=ChunkMetadata(
            heading_path=tuple(metadata.get("heading_path", [])),
            page_number=metadata.get("page_number"),
            token_count=metadata.get("token_count"),
            attributes=metadata.get("attributes", {}),
        ),
    )
