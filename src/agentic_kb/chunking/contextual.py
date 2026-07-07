"""Contextual chunk summaries used to improve retrieval indexing."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from dataclasses import replace
from typing import Protocol

from agentic_kb.config import load_env_file
from agentic_kb.schemas.chunks import Chunk, ChunkMetadata


CONTEXTUAL_SUMMARY_KEY = "contextual_summary"
DOCUMENT_TITLE_KEY = "document_title"


class LanguageModel(Protocol):
    """Minimal generation contract shared by provider adapters and tests."""

    @property
    def model_name(self) -> str:
        """Stable model identifier."""

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""


class ChunkContextualizer(Protocol):
    """Generates a short context summary for a chunk within a document."""

    @property
    def model_name(self) -> str:
        """Model or strategy name used for audit metadata."""

    @property
    def contextualizer_name(self) -> str:
        """Stable contextualizer identifier."""

    def generate_summary(
        self, *, document_text: str, chunk: Chunk, document_summary: str = ""
    ) -> str:
        """Return a short summary that makes the chunk easier to retrieve."""


class DummyFirstWordsContextualizer:
    """Deterministic local contextualizer for tests and baseline experiments."""

    contextualizer_name = "dummy-first-words"
    model_name = "dummy-first-words"

    def __init__(self, word_count: int = 10) -> None:
        if word_count <= 0:
            raise ValueError("word_count must be greater than zero")
        self._word_count = word_count

    def generate_summary(
        self, *, document_text: str, chunk: Chunk, document_summary: str = ""
    ) -> str:
        del chunk, document_summary
        return " ".join(document_text.split()[: self._word_count]).strip()


class LLMChunkContextualizer:
    """Generate Anthropic-style chunk context with the configured LLM.

    When a ``document_summary`` is provided, chunk prompts only receive
    the summary instead of the full document text.
    """

    contextualizer_name = "llm-contextual-summary"

    def __init__(
        self,
        model: LanguageModel | None = None,
        *,
        provider_factory: Callable[..., LanguageModel] | None = None,
    ) -> None:
        if model is None:
            model = _default_llm_model(provider_factory)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model.model_name

    def generate_summary(
        self, *, document_text: str, chunk: Chunk, document_summary: str = ""
    ) -> str:
        prompt = _chunk_prompt(document_summary=document_summary, chunk=chunk)
        return self._model.generate(prompt).strip()

    def generate_document_summary(self, document_text: str) -> str:
        """Produce a short document-level summary to seed chunk summaries."""
        prompt = (
            "用不超过300字总结以下文档的核心主题和关键要点。"
            "只返回摘要文本，不要其他内容。\n\n"
            f"{document_text.strip()}"
        )
        return self._model.generate(prompt).strip()


def contextualize_chunk(
    chunk: Chunk,
    *,
    summary: str,
    contextualizer: str,
    model_name: str,
) -> Chunk:
    """Return a copy of a chunk with contextual summary metadata attached."""

    clean_summary = summary.strip()
    if not clean_summary:
        return chunk

    attributes = {
        **chunk.metadata.attributes,
        CONTEXTUAL_SUMMARY_KEY: clean_summary,
    }
    metadata = replace(chunk.metadata, attributes=attributes)
    return replace(chunk, metadata=metadata)


def contextualize_chunks(
    *,
    document_text: str,
    chunks: list[Chunk],
    contextualizer: ChunkContextualizer,
) -> list[Chunk]:
    """Attach generated contextual summaries to chunks without changing source text.

    First generates a document-level summary, then reuses it for every chunk
    so the model does not re-read the whole document for each chunk.
    """
    # Stage 1 — document-level summary (500 chars max)
    doc_summary = ""
    if hasattr(contextualizer, "generate_document_summary"):
        doc_summary = contextualizer.generate_document_summary(document_text)
            

    # Stage 2 — per-chunk summary using the document summary as context
    results: list[Chunk] = []
    for chunk in chunks:
        summary = ""
        last_error = None
        for attempt in range(10):
            try:
                summary = contextualizer.generate_summary(
                    document_text=document_text,
                    chunk=chunk,
                    document_summary=doc_summary,
                )
                break
            except:
                time.sleep(wait)
        else:
            raise RuntimeError(
                f"contextualization failed after 10 attempts for chunk {chunk.id}"
            ) from last_error
        results.append(
            contextualize_chunk(
                chunk,
                summary=summary,
                contextualizer=contextualizer.contextualizer_name,
                model_name=contextualizer.model_name,
            )
        )
    return results


def text_for_indexing(chunk: Chunk) -> str:
    """Return text used for embedding/BM25 while preserving source chunk text.

    Prepends a short document title extracted from source_uri so that searches
    matching the filename can surface relevant chunks.
    """

    prefix_parts: list[str] = []
    source_uri = chunk.metadata.attributes.get("source_uri")
    if isinstance(source_uri, str):
        doc_title = _title_from_uri(source_uri)
        if doc_title:
            prefix_parts.append(doc_title)
    summary = chunk.metadata.attributes.get(CONTEXTUAL_SUMMARY_KEY)
    if isinstance(summary, str) and summary.strip():
        prefix_parts.append(summary.strip())

    if prefix_parts:
        return "\n\n".join(prefix_parts) + "\n\n" + chunk.text
    return chunk.text


def _title_from_uri(source_uri: str) -> str:
    """Extract a short document title from a source_uri for indexing.

    Returns the path after 'data/' (or the last 3 segments as fallback).
    """
    from urllib.parse import unquote, urlparse

    path = unquote(urlparse(source_uri).path)
    segments = [s for s in path.replace("\\", "/").split("/") if s]

    # Strip file extension from the last segment
    if segments:
        segments[-1] = Path(segments[-1]).stem

    try:
        idx = segments.index("data") + 1
        return "/".join(segments[idx:])
    except ValueError:
        tail = segments[-3:] if len(segments) >= 3 else segments
        return "/".join(tail)


def _chunk_prompt(*, document_summary: str, chunk: Chunk) -> str:
    """Chunk-level prompt using a pre-computed document summary."""
    if document_summary:
        return (
            "根据文档摘要，用中文为下面的chunk写一段简洁的检索上下文。"
            "必须严格控制在100字以内，只返回上下文描述，不要其他内容。\n\n"
            f"文档摘要:\n{document_summary}\n\n"
            f"Chunk文本:\n{chunk.text.strip()}"
        )
    return (
        "为下面的chunk写一段检索上下文，帮助后续检索时匹配到这段内容。\n"
        "只总结chunk的核心主题，不要复述chunk内容。\n"
        "必须严格控制在100字以内，超出100字的部分将被截断丢弃。\n"
        "只返回上下文描述，不要其他内容。\n"
        "使用与文档一致的语言。\n\n"
        f"文档全文:\n{chunk.text.strip()}"
    )


def _default_llm_model(provider_factory: Callable[..., LanguageModel] | None = None) -> LanguageModel:
    load_env_file(".env")
    if provider_factory is None:
        from agentic_kb.providers import Qwen

        provider_factory = Qwen
    return provider_factory(temperature=0, max_tokens=200, timeout=1200)