"""Retrieval package public API."""

from agentic_kb.retrieval.citations import CitationBuilder, CitationSource
from agentic_kb.retrieval.chunk_store import InMemoryChunkStore, SQLiteChunkStore
from agentic_kb.retrieval.context import ContextSelection, ContextSelector
from agentic_kb.retrieval.dense import DenseRetriever
from agentic_kb.retrieval.hybrid import HybridRetriever
from agentic_kb.retrieval.pipeline import RagPipeline, RagPipelineResult
from agentic_kb.retrieval.reranking import ApiReranker, CrossEncoderReranker
from agentic_kb.retrieval.sparse import SparseRetriever

__all__ = [
    "ApiReranker",
    "CitationBuilder",
    "CitationSource",
    "ContextSelection",
    "ContextSelector",
    "CrossEncoderReranker",
    "DenseRetriever",
    "HybridRetriever",
    "InMemoryChunkStore",
    "RagPipeline",
    "RagPipelineResult",
    "SQLiteChunkStore",
    "SparseRetriever",
]
