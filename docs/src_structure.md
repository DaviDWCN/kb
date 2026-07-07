# `src/agentic_kb` Structure

This document is the source map for the current codebase. It describes folders
that exist now and the boundary each one owns.

## `api/`

Public API-facing schemas live here. The project does not have HTTP routes yet,
but `api/schemas/` already defines request/response shapes for future document,
ingestion, and retrieval endpoints.

Keep HTTP contracts here instead of mixing them into internal domain schemas.

## `chunking/`

Turns parsed documents into smaller text chunks.

- `recursive.py` is the main structure-aware chunker.
- `constant_size.py` is the simpler fixed-size chunker.
- `schemas.py` contains chunking-specific config/result types.

Canonical chunk objects live in `schemas/chunks.py`, because chunks are shared
by embedding, retrieval, citation, and generation code.

## `config/`

Local configuration helpers.

Right now this contains `.env` loading through `python-dotenv`. CLI scripts call
this before creating model providers, so local API keys and endpoints can live
in `.env` while real shell environment variables still take priority.

## `embeddings/`

Embeds chunk text and writes vector records.

- `models.py` contains embedding model implementations such as `HashEmbeddingModel`.
- `service.py` validates and batches embedding calls.
- `indexer.py` converts chunks into `VectorRecord` objects and upserts them into a vector store.
- `schemas.py` contains embedding-only request/result shapes.

Provider-backed embedding adapters live in `providers/`, not here.

## `evaluation/`

Measures RAG quality.

- `judge.py` implements LLM-as-judge answer evaluation.
- `retrieval.py` implements retrieval metrics such as `hit@k`, `precision@k`, `recall@k`, `MRR@k`, and `NDCG@k`.

Answer quality and retrieval quality stay separate on purpose. A bad answer can
come from missing evidence, bad context selection, or the generation model, and
separate evaluators make that easier to diagnose.

## `generation/`

Builds final answers from selected context.

`answering.py` owns prompt construction, the no-context behavior, citation
attachment, and the final `Answer` object. It expects any model object with a
simple `generate(prompt: str) -> str` method.

## `metadata/`

Metadata-specific schemas.

Use this package for document/chunk metadata concepts that are not part of the
core shared chunk, document, vector, or search contracts.

## `parsing/`

Converts raw file bytes/text into `ParsedDocument`.

Current parsers cover plain text, Markdown, HTML, JSON, CSV, XLSX, DOCX, PDF,
and Docling-backed rich parsing. Parser classes also enforce size/output limits
and wrap unreadable files in parser-specific errors so ingestion can skip bad
files cleanly.

Chunkers should consume `ParsedDocument`; they should not parse raw files
themselves.

## `providers/`

Model API adapters.

The current adapters are:

- `Qwen` for answer/judge generation
- `DeepSeek` for answer/judge generation
- `BgeEmbedding` for embeddings

Provider classes hide OpenAI-compatible SDK calls and read model-specific env
vars such as `QWEN_API_KEY`, `BGE_API_KEY`, and `DEEPSEEK_API_KEY`.

## `retrieval/`

Search, reranking, context selection, citations, and pipeline orchestration.

- `dense.py` embeds a query and searches a vector store.
- `sparse.py` implements BM25-style lexical search.
- `hybrid.py` combines dense and sparse results.
- `reranking.py` wraps cross-encoder reranking.
- `context.py` selects ranked hits that fit a context budget.
- `citations.py` turns selected hits into citation sources.
- `chunk_store.py` hydrates retrieved vector IDs back into chunks, either in memory or SQLite.
- `pipeline.py` ties indexing, retrieval, reranking, context selection, and answer generation together.

Retrieval code depends on the vector-store contract, not a specific vector
database implementation.

## `schemas/`

Canonical shared domain objects.

Use this package for concepts that multiple subsystems depend on:

- documents and document IDs
- chunks and chunk metadata
- vector records and embeddings
- search hits and citations
- shared metadata

Do not put HTTP request/response shapes or storage row formats here.

## Schema Ownership Rules

Keep schemas close to the boundary they describe:

- `schemas/`: canonical domain objects shared by multiple packages
- `api/schemas/`: future HTTP request/response contracts
- `storage/schemas.py`: persistence row/serialized shapes
- package-local `schemas.py`: private shapes used by one subsystem only

Promote a schema to `schemas/` only when multiple packages need the same domain
concept.

## `storage/`

Storage-specific shapes.

This package is separate from vector storage. Use it for persistence records and
serialized database shapes that should not leak into higher-level pipeline code.

## `vector_store/`

Vector database contract and implementations.

- `base.py` defines the `VectorStore` interface.
- `hnsw.py` is the simple in-memory vector store used heavily in tests.
- `faiss.py` is an in-memory FAISS-backed vector store.
- `local_faiss.py` persists vector records in SQLite and rebuilds/writes a local FAISS index.

Pipeline and retrieval code should receive a `VectorStore` instance from the
outer wiring layer instead of importing a concrete implementation directly.
