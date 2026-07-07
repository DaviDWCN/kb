"""Constant-size token-window chunking."""

from agentic_kb.chunking.schemas import ChunkingConfig, ChunkingResult
from agentic_kb.parsing.schemas import ParsedDocument
from agentic_kb.schemas.chunks import Chunk, ChunkMetadata


class ConstantSizeChunker:
    """Chunk a parsed document into fixed-size whitespace-token windows."""

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        *,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> None:
        if config is None:
            if chunk_size is None:
                raise ValueError("chunk_size is required when config is not provided")
            config = ChunkingConfig(
                strategy="constant_size",
                max_tokens=chunk_size,
                overlap_tokens=0 if overlap is None else overlap,
                context_window_tokens=None,
                target_chunks_per_query=None,
            )
        self.config = config
        _validate_window(self.config.max_tokens, self.config.overlap_tokens)

    def chunk(self, document: ParsedDocument, *, document_id: str) -> ChunkingResult:
        """Convert document text into fixed-size chunks."""

        chunks = [
            Chunk(
                id=f"{document_id}:chunk-{index}",
                document_id=document_id,
                text=text,
                index=index,
                metadata=ChunkMetadata(
                    token_count=_token_count(text),
                    attributes={
                        "source_uri": document.source_uri,
                        "content_type": document.content_type,
                        "chunker": "constant_size",
                    },
                ),
            )
            for index, text in enumerate(
                _fixed_windows(document.text, self.config.max_tokens, self.config.overlap_tokens)
            )
        ]
        return ChunkingResult(document_id=document_id, chunks=chunks, config=self.config)


def _validate_window(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")


def _fixed_windows(text: str, chunk_size: int, overlap: int) -> list[str]:
    tokens = text.split()
    if not tokens:
        return []

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap
    while start < len(tokens):
        if start > 0 and len(tokens) - start <= overlap:
            break
        chunks.append(" ".join(tokens[start : start + chunk_size]))
        start += step
    return chunks


def _token_count(text: str) -> int:
    """Estimate token count by characters (~1 tok per CJK char, ~4 chars per English tok)."""
    import re
    cjk = len(re.findall(r'[\u4e00-\u9fff]', text))
    other = max(len(text) - cjk, 0)
    return cjk + (other + 3) // 4
