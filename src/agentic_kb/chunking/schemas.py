"""Chunking-specific request/result data shapes."""

from dataclasses import dataclass, field

from agentic_kb.schemas.chunks import Chunk


DEFAULT_CONTEXT_WINDOW_TOKENS = 32_000
DEFAULT_RESERVED_PROMPT_TOKENS = 4_000
DEFAULT_RESERVED_ANSWER_TOKENS = 4_000
DEFAULT_TARGET_CHUNKS_PER_QUERY = 8
DEFAULT_MAX_CHUNK_TOKENS = 2_000
DEFAULT_OVERLAP_TOKENS = 200


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration shared by chunker implementations.

    max_tokens is the hard cap for one chunk. When context-window fields are
    provided, RecursiveChunker derives an effective max_tokens value from the
    retrieval budget before splitting.
    """

    strategy: str
    max_tokens: int = DEFAULT_MAX_CHUNK_TOKENS
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS
    min_tokens: int | None = None
    merge_target_tokens: int | None = None
    context_window_tokens: int | None = DEFAULT_CONTEXT_WINDOW_TOKENS
    reserved_prompt_tokens: int = DEFAULT_RESERVED_PROMPT_TOKENS
    reserved_answer_tokens: int = DEFAULT_RESERVED_ANSWER_TOKENS
    target_chunks_per_query: int | None = DEFAULT_TARGET_CHUNKS_PER_QUERY


@dataclass(frozen=True)
class ChunkBoundary:
    """Source text boundary for a produced chunk when offsets are available."""

    start: int
    end: int
    heading_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChunkingResult:
    """Result returned by chunkers after turning parsed content into chunks."""

    document_id: str
    chunks: list[Chunk]
    boundaries: list[ChunkBoundary] = field(default_factory=list)
    config: ChunkingConfig | None = None
