"""Local persistent FAISS vector store backed by SQLite records."""

import json
import sqlite3
from pathlib import Path
from typing import Any

from agentic_kb.schemas.vectors import Embedding, Metadata, SearchResult, VectorRecord
from agentic_kb.vector_store.base import VectorStore
from agentic_kb.vector_store.faiss import (
    _load_faiss,
    _load_numpy,
    _normalize_rows,
)


_SCHEMA_VERSION = 1


class LocalFaissVectorStore(VectorStore):
    """Persist vector records locally and rebuild a FAISS index from SQLite."""

    def __init__(
        self,
        store_dir: str | Path,
        *,
        dimensions: int,
        model_name: str = "unknown",
    ) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")

        self._store_dir = Path(store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._store_dir / "rag.sqlite"
        self._index_path = self._store_dir / "faiss.index"
        self._manifest_path = self._store_dir / "manifest.json"
        self._dimensions = dimensions
        self._model_name = model_name
        self._faiss = _load_faiss()
        self._np = _load_numpy()
        self._connection = sqlite3.connect(self._db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._ensure_schema()
        self._validate_or_write_manifest()
        self._records = self._load_records()
        self._chunk_ids = list(self._records.keys())
        self._index = self._load_or_rebuild_index()

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def upsert(self, record: VectorRecord) -> None:
        self._validate_embedding(record.embedding)
        self._records[record.chunk_id] = record.document_id
        self._save_record(record)
        self._rebuild_index()

    def upsert_many(self, records: list[VectorRecord]) -> None:
        """Batch upsert: persist to SQLite and incrementally grow the FAISS index.

        Only truly new chunk_ids are appended to the in-memory index — existing
        vectors are not re-loaded or re-added. This keeps large-scale build
        throughput linear rather than O(n²).
        """
        if not records:
            return

        # Separate truly new records from updates (same chunk_id already indexed).
        new_records = [r for r in records if r.chunk_id not in self._records]

        for record in records:
            self._validate_embedding(record.embedding)
            self._records[record.chunk_id] = record.document_id

        with self._connection:
            self._connection.executemany(
                """
                INSERT OR REPLACE INTO vector_records (
                    chunk_id, document_id, embedding_json
                )
                VALUES (?, ?, ?)
                """,
                [
                    (record.chunk_id, record.document_id, json.dumps(record.embedding))
                    for record in records
                ],
            )

        # Incremental add: append new vectors to the existing FAISS index.
        # Avoids a full rebuild which would reload *all* embeddings from SQLite.
        if new_records:
            new_embeddings = [r.embedding for r in new_records]
            self._index.add(self._as_matrix(new_embeddings))
            self._chunk_ids.extend(r.chunk_id for r in new_records)
            self._write_index()

    def delete(self, chunk_id: str) -> None:
        if chunk_id not in self._records:
            return

        self._records.pop(chunk_id)
        with self._connection:
            self._connection.execute(
                "DELETE FROM vector_records WHERE chunk_id = ?",
                (chunk_id,),
            )
        self._rebuild_index()

    def update(
        self,
        chunk_id: str,
        *,
        embedding: Embedding | None = None,
        metadata: Metadata | None = None,
    ) -> None:
        document_id = self._records[chunk_id]
        next_embedding = embedding if embedding is not None else self._load_embedding(chunk_id)
        self._validate_embedding(next_embedding)
        self._save_record(
            VectorRecord(
                chunk_id=chunk_id,
                document_id=document_id,
                embedding=next_embedding,
                metadata={},
            )
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

        query_vector = self._as_matrix([query])
        distances, indices = self._index.search(query_vector, len(self._chunk_ids))

        results: list[SearchResult] = []
        for score, index in zip(distances[0], indices[0]):
            if index < 0:
                continue
            chunk_id = self._chunk_ids[int(index)]
            results.append(
                SearchResult(
                    record=VectorRecord(
                        chunk_id=chunk_id,
                        document_id=self._records[chunk_id],
                        embedding=[],
                        metadata={},
                    ),
                    score=float(score),
                )
            )
            if len(results) == k:
                break
        return results

    def get(self, chunk_id: str) -> VectorRecord | None:
        document_id = self._records.get(chunk_id)
        if document_id is None:
            return None
        embedding = self._load_embedding(chunk_id)
        return VectorRecord(
            chunk_id=chunk_id,
            document_id=document_id,
            embedding=embedding,
            metadata={},
        )

    def __len__(self) -> int:
        return len(self._records)

    def _ensure_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vector_records (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    embedding_json TEXT NOT NULL
                )
                """
            )

    def _validate_or_write_manifest(self) -> None:
        if self._manifest_path.exists():
            manifest = json.loads(self._manifest_path.read_text())
            if manifest.get("dimensions") != self._dimensions:
                raise ValueError(
                    f"stored dimensions {manifest.get('dimensions')} do not match {self._dimensions}"
                )
            if manifest.get("schema_version") != _SCHEMA_VERSION:
                raise ValueError(
                    f"stored schema version {manifest.get('schema_version')} is not supported"
                )
            stored_model = manifest.get("model_name")
            if self._model_name != "unknown" and stored_model != self._model_name:
                raise ValueError(f"stored model {stored_model!r} does not match {self._model_name!r}")
            return

        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "dimensions": self._dimensions,
            "model_name": self._model_name,
        }
        self._manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    def _load_records(self) -> dict[str, str]:
        rows = self._connection.execute(
            "SELECT chunk_id, document_id FROM vector_records ORDER BY rowid"
        ).fetchall()
        return {row["chunk_id"]: row["document_id"] for row in rows}

    def _load_embedding(self, chunk_id: str) -> Embedding:
        row = self._connection.execute(
            "SELECT embedding_json FROM vector_records WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return []
        return [float(value) for value in json.loads(row["embedding_json"])]

    def _load_embeddings_batch(self, chunk_ids: list[str]) -> list[Embedding]:
        """Load embeddings from SQLite in chunk_id order, returning numpy-ready float lists."""
        embedding_map: dict[str, Embedding] = {}
        # SQLite limits IN clause to 999 bound parameters; chunk the query.
        chunk_size = 900
        for start in range(0, len(chunk_ids), chunk_size):
            batch = chunk_ids[start:start + chunk_size]
            placeholders = ",".join("?" * len(batch))
            rows = self._connection.execute(
                f"SELECT chunk_id, embedding_json FROM vector_records WHERE chunk_id IN ({placeholders})",
                batch,
            ).fetchall()
            for row in rows:
                embedding_map[row["chunk_id"]] = [
                    float(value) for value in json.loads(row["embedding_json"])
                ]
        return [embedding_map[chunk_id] for chunk_id in chunk_ids]

    def _save_record(self, record: VectorRecord) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO vector_records (
                    chunk_id, document_id, embedding_json
                )
                VALUES (?, ?, ?)
                """,
                (record.chunk_id, record.document_id, json.dumps(record.embedding)),
            )

    def _load_or_rebuild_index(self) -> Any:
        """Load FAISS index from disk if it matches SQLite records, otherwise rebuild."""
        if self._index_path.exists() and self._index_path.stat().st_size > 0:
            try:
                index = self._faiss.read_index(str(self._index_path))
                if index.ntotal == len(self._chunk_ids):
                    return index
            except Exception:
                pass
        return self._build_index_from_sqlite()

    def _build_index_from_sqlite(self) -> Any:
        """Rebuild FAISS index from scratch using SQLite-stored embeddings."""
        self._index = self._new_index()
        if self._chunk_ids:
            embeddings = self._load_embeddings_batch(self._chunk_ids)
            self._index.add(self._as_matrix(embeddings))
        self._write_index()
        return self._index

    def _rebuild_index(self) -> None:
        """Full index rebuild — used by upsert (single) and delete."""
        self._chunk_ids = list(self._records.keys())
        self._index = self._build_index_from_sqlite()

    def _write_index(self) -> None:
        write_index = getattr(self._faiss, "write_index", None)
        if callable(write_index):
            write_index(self._index, str(self._index_path))
            return
        self._index_path.touch()

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
