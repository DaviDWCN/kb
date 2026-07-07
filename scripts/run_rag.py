#!/usr/bin/env python3

# Build and query a local FAISS-backed RAG store.
#
# Usage:
#   python scripts/run_rag.py build --embedding bge
#   python scripts/run_rag.py query "test" --embedding bge --answer-model deepseek

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the src/ package root is on sys.path.
_PROJECT = Path(__file__).resolve().parents[1]
_src = _PROJECT / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import argparse
import json
import os
import re
import threading
from urllib.parse import unquote
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from agentic_kb.chunking import (
    ChunkingConfig,
    DummyFirstWordsContextualizer,
    LLMChunkContextualizer,
    RecursiveChunker,
    contextualize_chunks,
)
from agentic_kb.config import load_env_file
from agentic_kb.embeddings import EmbeddingService, HashEmbeddingModel
from agentic_kb.generation import AnswerGenerator
from agentic_kb.parsing import (
    ParserDependencyError,
    ParserLimitError,
    ParserReadError,
    UnsupportedContentTypeError,
    parser_for_path,
)
from agentic_kb.retrieval import (
    ApiReranker,
    CitationBuilder,
    ContextSelector,
    CrossEncoderReranker,
    RagPipeline,
    SQLiteChunkStore,
)
from agentic_kb.vector_store import LocalFaissVectorStore


DEFAULT_DATA_DIR = _PROJECT / "data"
DEFAULT_STORE_DIR = _PROJECT / ".local_store"
DEFAULT_HASH_DIMENSIONS = 64
DEFAULT_BGE_DIMENSIONS = 1024
class RetrievedContextModel:
    """Tiny local answer model that echoes retrieved context instead of calling an API."""

    def generate(self, prompt: str) -> str:
        if prompt.startswith("No document context was selected"):
            return (
                "No chunks were retrieved or selected. Possible explanations: "
                "the local vector DB is empty, the question does not match the stored documents, "
                "or the context budget filtered out every chunk."
            )

        context = _between(prompt, "Context:\n", "\n\nAnswer:")
        if not context.strip():
            return "Retrieved context was empty."
        return "Based on the retrieved document chunks:\n" + _clip(context.strip(), 1_500)


def main(argv: list[str] | None = None) -> int:
    load_env_file(_PROJECT / ".env")
    parser = _arg_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "build":
            return build_vector_db(args)
        if args.command == "convert":
            return run_convert(args)
        if args.command == "query":
            return query_vector_db(args)
        if args.command == "server":
            return run_server(args)
    except ImportError as error:
        raise SystemExit(
            f"{error}\nInstall the missing optional dependency, for example: "
            "pip install -e '.[vector,providers,reranking]'"
        ) from error

    parser.print_help()
    return 1


def build_vector_db(args: argparse.Namespace) -> int:
    data_dir = args.data_dir.resolve()
    store_dir = args.store_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    # Build always reads from intermediate Markdown files.
    source_dir = data_dir / ".intermediate"
    if not source_dir.exists():
        raise SystemExit(
            f"No intermediate directory at {source_dir}. "
            "Run `python scripts/run_rag.py convert` first."
        )
    files = list(_iter_data_files(source_dir))
    if not files:
        raise SystemExit(
            f"No .md files found in {source_dir}. "
            "Run `python scripts/run_rag.py convert` first."
        )

    model = _embedding_model(args.embedding, _build_dimensions(args))
    pipeline = _pipeline_for(store_dir, model, answer_model=RetrievedContextModel(), args=args)
    chunker = RecursiveChunker(
        ChunkingConfig(
            strategy="recursive",
            max_tokens=args.max_tokens,
            overlap_tokens=args.overlap_tokens,
            merge_target_tokens=args.merge_target_tokens,
        )
    )
    contextualizer = _contextualizer_for_build(args.contextualization)

    workers = max(1, getattr(args, "workers", 1))
    print(f"Build started: {len(files)} files, {workers} workers", flush=True)

    if workers == 1:
        indexed_files, indexed_chunks, skipped = _build_sequential(
            files, source_dir, pipeline, chunker, contextualizer,
        )
    else:
        indexed_files, indexed_chunks, skipped = _build_concurrent(
            files, source_dir, pipeline, chunker, contextualizer, workers,
        )

    print()
    print(f"Store: {store_dir}")
    print(f"Indexed: {indexed_files} files, {indexed_chunks} chunks")
    if skipped:
        print(f"Skipped: {len(skipped)} files")
        for path, reason in skipped:
            print(f"- {path.relative_to(data_dir)}: {reason}")
    return 0


def _build_sequential(
    files: list[Path],
    data_dir: Path,
    pipeline: RagPipeline,
    chunker: RecursiveChunker,
    contextualizer,
) -> tuple[int, int, list[tuple[Path, str]]]:
    """Original single-threaded build — parse, chunk, contextualize, index per file."""
    indexed_files = 0
    indexed_chunks = 0
    skipped: list[tuple[Path, str]] = []

    for path in files:
        chunks = _process_file(path, data_dir, chunker, contextualizer)
        if isinstance(chunks, str):
            skipped.append((path, chunks))
            continue
        pipeline.index_chunks(chunks)
        indexed_files += 1
        indexed_chunks += len(chunks)
        print(f"Indexed {path.relative_to(data_dir)} ({len(chunks)} chunks)")

    return indexed_files, indexed_chunks, skipped


def _build_concurrent(
    files: list[Path],
    data_dir: Path,
    pipeline: RagPipeline,
    chunker: RecursiveChunker,
    contextualizer,
    workers: int,
) -> tuple[int, int, list[tuple[Path, str]]]:
    """High-concurrency build: parse+chunk in threads, flush to vector store in batches.

    Chunks are accumulated in memory only up to *batch_size* files' results,
    then embedded and persisted together. This bounds peak memory regardless of
    total file count, while still gaining embedding throughput from batching.
    """
    batch_size = 20
    buffer: list = []
    skipped: list[tuple[Path, str]] = []
    skip_lock = threading.Lock()
    completed = 0
    total = len(files)
    total_chunks = 0

    # Force stdout to flush immediately so C-level stderr warnings
    # don't hide python print output behind buffering.
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_path = {
            executor.submit(_process_file, path, data_dir, chunker, contextualizer): path
            for path in files
        }

        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                result = future.result()
                if isinstance(result, str):
                    with skip_lock:
                        skipped.append((path, result))
                else:
                    buffer.extend(result)
                    completed += 1
                    print(f"[{completed}/{total}] Parsed {path.relative_to(data_dir)} ({len(result)} chunks)")
            except Exception as error:
                with skip_lock:
                    skipped.append((path, str(error)))

            # Flush buffer every batch_size files.
            if completed > 0 and completed % batch_size == 0 and buffer:
                total_chunks += len(buffer)
                print(f"  Flushing {len(buffer)} chunks from last {batch_size} files...", flush=True)
                pipeline.index_chunks(buffer)
                buffer = []

    # Flush remaining chunks.
    if buffer:
        total_chunks += len(buffer)
        pipeline.index_chunks(buffer)

    return completed, total_chunks, skipped




def _doc_to_markdown(document, source_path: Path) -> str:
    """Convert a ParsedDocument to a readable Markdown string.

    Preserves section structure and front-matter metadata for auditability.
    """
    lines: list[str] = []
    # Front-matter first so MarkdownParser can extract it before headings.
    lines.append(f"> **source_uri**: {document.source_uri}")
    lines.append(f"> **content_type**: {document.content_type}")
    for key, value in sorted(document.metadata.items()):
        # file_mtime must survive the convert→build roundtrip so the
        # MarkdownParser can recover the original source file's date.
        if key in {"parser", "source_format", "page_count", "ocr_page_count", "ocr_attempted_page_count"}:
            continue
        lines.append(f"> **{key}**: {value}")
    lines.append("")
    lines.append(f"# {source_path.stem}")
    lines.append("")

    for section in document.sections:
        heading = section.title or f"Section {section.index + 1}"
        lines.append(f"## {heading}")
        lines.append("")
        if section.text.strip():
            lines.append(section.text.strip())
        lines.append("")

    return "\n".join(lines)


def _write_intermediate_markdown(
    document,
    source_path: Path,
    data_dir: Path,
    markdown_dir: Path,
) -> None:
    """Persist a parsed document as a Markdown file mirroring the source tree."""
    relative = source_path.relative_to(data_dir)
    md_path = markdown_dir / relative.with_suffix(".md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_doc_to_markdown(document, source_path), encoding="utf-8")


def run_convert(args: argparse.Namespace) -> int:
    """Parse all source files and save as Markdown to data/.intermediate/.

    No chunking, embedding, or indexing — pure format conversion.
    """
    data_dir = args.data_dir.resolve()
    markdown_dir = data_dir / ".intermediate"
    files = [f for f in _iter_data_files(data_dir) if markdown_dir not in f.parents and f != markdown_dir]
    if not files:
        print(f"No files found in {data_dir}.")
        return 0

    workers = max(1, getattr(args, "workers", 1))
    print(f"Converting {len(files)} files to Markdown ({workers} workers)...", flush=True)

    def _convert_one(path: Path) -> tuple[Path, str | None]:
        try:
            document = _parse_file(path)
        except Exception as error:
            return path, str(error)
        try:
            _write_intermediate_markdown(document, path, data_dir, markdown_dir)
        except OSError as error:
            return path, str(error)
        return path, None

    converted = 0
    failed: list[tuple[Path, str]] = []

    if workers == 1:
        for path in files:
            _, error = _convert_one(path)
            if error:
                failed.append((path, error))
            else:
                converted += 1
                print(f"  {path.relative_to(data_dir)}")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_path = {executor.submit(_convert_one, path): path for path in files}
            for future in as_completed(future_to_path):
                path, error = future.result()
                if error:
                    failed.append((path, error))
                else:
                    converted += 1
                    print(f"  [{converted}/{len(files)}] {path.relative_to(data_dir)}")

    print(f"\nDone: {converted} files → {markdown_dir}")
    if failed:
        print(f"Failed: {len(failed)} files")
        for path, reason in failed:
            print(f"  - {path.relative_to(data_dir)}: {reason}")
    return 0


def _process_file(
    path: Path,
    data_dir: Path,
    chunker: RecursiveChunker,
    contextualizer,
) -> list | str:
    """Parse, chunk, and optionally contextualize a single file.

    Returns a list of chunks on success, or an error string on failure.
    This function is designed to be called from worker threads.
    """
    try:
        document = _parse_file(path)
        if len(document.text.strip()) < 50:
            return "parsed successfully but produced less than 50 characters of text"
        document_id = _document_id_for(path, data_dir)
        chunks = [
            chunk
            for chunk in chunker.chunk(document, document_id=document_id).chunks
            if chunk.text.strip()
        ]
        if not chunks:
            return "parsed successfully but produced no text chunks"

        # Inject file_mtime into chunk attributes for vector-level filtering only.
        # Not routed through document→chunk metadata propagation to avoid clutter.
        if "file_mtime" in document.metadata:
            for chunk in chunks:
                chunk.metadata.attributes["file_mtime"] = document.metadata["file_mtime"]

        if contextualizer is not None:
            chunks = contextualize_chunks(
                document_text=document.text,
                chunks=chunks,
                contextualizer=contextualizer,
            )
        return chunks
    except (
        OSError,
        ParserDependencyError,
        ParserLimitError,
        ParserReadError,
        UnsupportedContentTypeError,
    ) as error:
        return str(error)


def query_vector_db(args: argparse.Namespace) -> int:
    store_dir = args.store_dir.resolve()
    if not (store_dir / "manifest.json").exists():
        raise SystemExit(f"No local vector DB found at {store_dir}. Run the build command first.")

    dimensions = _stored_dimensions(store_dir)
    model = _embedding_model(args.embedding, dimensions)
    answer_model = _answer_model(args.answer_model)
    pipeline = _pipeline_for(store_dir, model, answer_model=answer_model, args=args)

    keywords = _extract_keywords(args.question, answer_model)
    if args.no_expand:
        queries = None
    else:
        hyde = _hyde_query(args.question, answer_model)
        term_mapping = _infer_term_variants(args.question, answer_model)
        raw_variants = _expand_query_with_mapping(args.question, term_mapping, answer_model) or []
        variants = raw_variants or _expand_query(args.question, answer_model) or []
        queries = ([hyde] if hyde else []) + variants
        if not queries:
            queries = None
    result = pipeline.answer(args.question, k=args.top_k, expanded_query=queries, sparse_query=keywords)
    if args.output is not None:
        write_json_output(args.output, query_result_to_dict(result))

    print("Answer")
    print("------")
    print(result.answer.text)
    if args.no_chunks:
        return 0
    print()
    print("Retrieved Chunks")
    print("----------------")
    hits = result.reranked[:args.top_k] if result.reranked else (result.context.selected or result.candidates)
    if not hits:
        print("No chunks retrieved.")
        return 0

    for index, hit in enumerate(hits[:args.top_k], start=1):
        source_uri = hit.chunk.metadata.attributes.get("source_uri", "unknown source")
        source_uri = unquote(source_uri)
        print(f"[{index}] {hit.chunk.id} score={hit.score:.4f} source={source_uri}")
        print(_clip(hit.chunk.text.strip(), args.snippet_chars))
        print()
    return 0


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build/query a local AgenticKB vector DB.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Parse files in data/ and build the vector DB.")
    _add_common_args(build)
    build.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of parallel workers for file parsing (default: 1, sequential).",
    )
    build.add_argument("--max-tokens", type=int, default=750)
    build.add_argument("--overlap-tokens", type=int, default=60)
    build.add_argument("--merge-target-tokens", type=int, default=300)
    build.add_argument(
        "--contextualization",
        choices=("none", "dummy", "llm"),
        default=os.getenv("RAG_CONTEXTUALIZATION", "none"),
        help="Optionally add contextual summaries to chunks before indexing.",
    )

    convert = subparsers.add_parser("convert", help="Parse all files in data/ and save as Markdown to data/.intermediate/.")
    _add_common_args(convert)
    convert.add_argument("--workers", "-w", type=int, default=1)

    query = subparsers.add_parser("query", help="Retrieve chunks from an existing vector DB.")
    _add_common_args(query)
    query.add_argument("question")
    query.add_argument("--answer-model", choices=("echo", "qwen", "deepseek"), default=os.getenv("RAG_ANSWER_MODEL", "echo"))
    query.add_argument("--top-k", type=int, default=20)
    query.add_argument("--context-tokens", type=int, default=20000)
    query.add_argument(
        "--reranker",
        default=os.getenv("RAG_RERANKER"),
        help="Optional sentence-transformers cross-encoder model name for reranking.",
    )
    query.add_argument("--output", type=Path, help="Optional path to write query result JSON.")
    query.add_argument("--no-chunks", action="store_true", help="Skip printing retrieved chunks.")
    query.add_argument("--no-expand", action="store_true", help="Disable LLM query expansion.")
    query.add_argument("--snippet-chars", type=int, default=700)
    query.add_argument("--dense-only", action="store_true", help="Skip sparse BM25 retrieval, use FAISS dense search only.")

    server = subparsers.add_parser("server", help="Start a local RAG API server.")
    _add_common_args(server)
    server.add_argument("--port", type=int, default=8080)
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--top-k", type=int, default=20)
    server.add_argument("--answer-model", choices=("echo", "qwen", "deepseek"), default=os.getenv("RAG_ANSWER_MODEL", "echo"))
    server.add_argument("--context-tokens", type=int, default=20000)
    server.add_argument("--reranker", default=os.getenv("RAG_RERANKER"))
    server.add_argument("--dense-only", action="store_true", help="Skip sparse BM25, use FAISS dense-only retrieval.")
    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    parser.add_argument("--embedding", choices=("hash", "bge"), default=os.getenv("RAG_EMBEDDING", "hash"))
    parser.add_argument("--dimensions", type=int)


def _pipeline_for(
    store_dir: Path,
    model,
    *,
    answer_model,
    args: argparse.Namespace,
) -> RagPipeline:
    vector_store = LocalFaissVectorStore(
        store_dir,
        dimensions=model.dimensions,
        model_name=model.model_name,
    )
    return RagPipeline(
        embedding_service=EmbeddingService(model),
        vector_store=vector_store,
        answer_generator=AnswerGenerator(answer_model, CitationBuilder()),
        context_selector=ContextSelector(max_tokens=getattr(args, "context_tokens", 2_500)),
        reranker=_reranker_for(getattr(args, "reranker", None)),
        chunk_store=SQLiteChunkStore(store_dir / "rag.sqlite"),
        search_k=getattr(args, "top_k", 5),
        search_width=getattr(args, "top_k", 5) * 3,
        dense_only=bool(getattr(args, "dense_only", False)),
    )


def _embedding_model(name: str, dimensions: int):
    if name == "hash":
        return HashEmbeddingModel(dimensions=dimensions)
    if name == "bge":
        from agentic_kb.providers import BgeEmbedding

        return BgeEmbedding(dimensions=dimensions)
    raise ValueError(f"unsupported embedding model: {name}")


def _answer_model(name: str):
    if name == "echo":
        return RetrievedContextModel()
    if name == "qwen":
        from agentic_kb.providers import Qwen

        return Qwen(temperature=0.7, max_tokens=2000)
    if name == "deepseek":
        from agentic_kb.providers import DeepSeek

        return DeepSeek(temperature=0.7, max_tokens=2000, timeout=300)
    raise ValueError(f"unsupported answer model: {name}")


def _contextualizer_for_build(name: str):
    if name == "none":
        return None
    if name == "dummy":
        return DummyFirstWordsContextualizer()
    if name == "llm":
        return LLMChunkContextualizer()
    raise ValueError(f"unsupported contextualization mode: {name}")


def _reranker_for(model_name: str | None):
    if model_name is None:
        return None
    cleaned_name = model_name.strip()
    if not cleaned_name:
        return None
    # API-based reranker when RERANKER_BASE_URL is configured
    if os.getenv("RERANKER_BASE_URL"):
        return ApiReranker(cleaned_name)
    return CrossEncoderReranker.from_sentence_transformers(cleaned_name)


def _build_dimensions(args: argparse.Namespace) -> int:
    if args.dimensions is not None:
        return args.dimensions
    if args.embedding == "bge":
        return DEFAULT_BGE_DIMENSIONS
    return DEFAULT_HASH_DIMENSIONS


def _stored_dimensions(store_dir: Path) -> int:
    manifest = json.loads((store_dir / "manifest.json").read_text())
    dimensions = manifest.get("dimensions")
    if not isinstance(dimensions, int):
        raise SystemExit(f"Invalid manifest at {store_dir / 'manifest.json'}: missing dimensions")
    return dimensions


def query_result_to_dict(result) -> dict[str, object]:
    """Convert a RAG pipeline result into JSON-serializable data."""
    return {
        "answer": {
            "query": result.answer.query,
            "text": result.answer.text,
            "used_chunk_ids": list(result.answer.used_chunk_ids),
            "citations": [_citation_to_dict(citation) for citation in result.answer.citations],
        },
        "context": {
            "token_count": result.context.token_count,
            "max_tokens": result.context.max_tokens,
            "selected_count": len(result.context.selected),
            "omitted_count": len(result.context.omitted),
        },
        "selected_chunks": [_hit_to_dict(hit) for hit in result.context.selected],
        "omitted_chunks": [_hit_to_dict(hit) for hit in result.context.omitted],
        "reranked": [_hit_to_dict(hit) for hit in result.reranked],
        "candidates": [_hit_to_dict(hit) for hit in result.candidates],
    }


def write_json_output(path: Path, payload: dict[str, object]) -> None:
    """Write JSON output, creating parent directories when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _hit_to_dict(hit) -> dict[str, object]:
    chunk = hit.chunk
    return {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "text": chunk.text,
        "chunk_index": chunk.index,
        "score": hit.score,
        "citation": _search_citation_to_dict(hit.citation),
        "highlights": list(hit.highlights),
        "metadata": _chunk_metadata_to_dict(chunk.metadata),
    }


def _chunk_metadata_to_dict(metadata) -> dict[str, object]:
    return {
        "heading_path": list(metadata.heading_path),
        "page_number": metadata.page_number,
        "token_count": metadata.token_count,
        **dict(metadata.attributes),
    }


def _citation_to_dict(citation) -> dict[str, object]:
    return {
        "document_id": citation.document_id,
        "chunk_id": citation.chunk_id,
        "score": citation.score,
        "source_uri": citation.source_uri,
        "heading_path": list(citation.heading_path),
        "page_number": citation.page_number,
        "chunk_index": citation.chunk_index,
    }


def _search_citation_to_dict(citation) -> dict[str, object] | None:
    if citation is None:
        return None
    span = getattr(citation, "span", None)
    return {
        "document_id": citation.document_id,
        "chunk_id": citation.chunk_id,
        "score": citation.score,
        "span": list(span) if span is not None else None,
    }


def _parse_file(path: Path):
    parser, content_type = parser_for_path(path)
    if parser is None or content_type is None:
        raise UnsupportedContentTypeError(f"unsupported file suffix: {path.suffix or '<none>'}")

    # Read bytes and let each parser decide how to decode or inspect the format.
    document = parser.parse(
        path.read_bytes(),
        source_uri=path.resolve().as_uri(),
        content_type=content_type,
    )
    # Attach source file mtime for citation traceability.
    # For intermediate .md files, the parser extracts it from front-matter.
    if "file_mtime" not in document.metadata:
        document.metadata["file_mtime"] = _format_mtime(path.stat().st_mtime)
    return document

def _iter_data_files(data_dir: Path):
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith(".") or path.name.startswith("~$"):
            continue
        # Skip spreadsheets — structured tabular data belongs in SQL, not vectors.
        if path.suffix.lower() in {".xlsx", ".xls", ".csv"}:
            continue
        # Skip print template files.
        if "打印模板" in str(path):
            continue
        yield path


def _document_id_for(path: Path, data_dir: Path) -> str:
    relative = path.relative_to(data_dir).as_posix()
    document_id = re.sub(r"[^A-Za-z0-9._-]+", "-", relative.replace("/", "__"))
    return document_id.strip("-") or "document"


def _format_mtime(timestamp: float) -> str:
    """Convert a POSIX timestamp to a human-readable UTC datetime string."""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")


def _between(text: str, start: str, end: str) -> str:
    _, found, remainder = text.partition(start)
    if not found:
        return ""
    value, _, _ = remainder.partition(end)
    return value


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _infer_term_variants(query: str, model) -> str | None:
    """Let LLM infer alternative names for domain-specific terms in the query.

    Returns a summary string like "续保自动转险种 → 跨险种续保, 续保险种转换",
    or None if no term mapping is needed.
    """
    if not query.strip():
        return None
    try:
        response = model.generate(
            "你是一个保险科技领域的术语专家。用户的搜索查询中可能包含一些术语，"
            "这些术语在内部不同系统、不同部门可能有不同叫法。\n"
            "请推断这些术语可能的其他叫法，用「原始术语 → 替代叫法」格式输出。\n"
            "只输出确实存在替代叫法的术语，不要硬凑。每行一组映射。\n\n"
            "例如：\n"
            "续保自动转险种 → 跨险种续保, 续保险种转换\n\n"
            f"查询: {query}\n\n术语映射:"
        ).strip()
    except Exception:
        return None
    if not response or len(response) > 300:
        return None
    return response


def _expand_query_with_mapping(query: str, term_mapping: str | None, model) -> list[str] | None:
    """Generate retrieval queries informed by term equivalences."""
    if not query.strip():
        return None
    try:
        if term_mapping:
            prompt = (
                "用户在企业知识库中搜索。已知以下术语对应关系：\n"
                f"{term_mapping}\n\n"
                "用这些替代叫法生成2-3个检索查询，"
                "模拟不同用户/不同部门的提问方式。\n"
                "每行一个查询，直接输出，不要编号、不要思考过程、不要解释。\n\n"
                f"用户问题: {query}"
            )
        else:
            prompt = (
                "用户在企业知识库中搜索。请做两件事：\n"
                "1. 推断用户真正想找什么类型的文档（需求文档？方案？操作手册？）\n"
                "2. 设想这类文档在公司内部可能用什么标题或写法，生成2个不同措辞的查询\n\n"
                "注意：公司内部同一件事在不同部门/不同时期叫法不同，"
                "要从业务逻辑推断可能的文档命名方式。\n\n"
                "每行一个查询，直接输出，不要编号、不要思考过程、不要解释。"
                f"用户问题: {query}"
            )
        response = model.generate(prompt).strip()
    except Exception:
        return None
    if not response or len(response) > 500:
        return None
    variants = [line.strip() for line in response.splitlines() if line.strip() and line.strip() != query]
    if not variants:
        return None
    print(f"query variants: {variants}")
    return variants


def _expand_query(query: str, model) -> list[str] | None:
    """Fallback: original expand without term mapping."""
    return _expand_query_with_mapping(query, None, model)


def _hyde_query(query: str, model) -> str | None:
    """Use LLM to generate a hypothetical answer, then search with it.

    A fake answer is semantically closer to real document text than the question itself.
    """
    if not query.strip():
        return None
    try:
        response = model.generate(
            "用户在华泰保险内部知识库搜索。先推断这个问题对应的答案最可能出现在哪种文档里，"
            "然后用该类文档的典型写法生成一段假答案。假答案要具体，包含系统名和项目编号。"
            "不超过250字，直接输出假答案，不要前缀。\n\n"
            f"用户问题: {query}"
        ).strip()
    except Exception:
        return None
    if not response or len(response) > 600:
        return None
    return response


def _extract_keywords(query: str, model) -> str | None:
    """Use LLM to extract dense keywords for BM25 sparse retrieval."""
    if not query.strip():
        return None
    try:
        keywords = model.generate(
            "用户在华泰保险内部知识库搜索。提取用于全文检索的关键词，用空格分隔。\n"
            "不仅提取问题中的词，还要推测相关文档可能使用的其他术语。\n"
            "公司和系统名要同时给出简称和全称。\n"
            "只输出关键词，不要解释。\n\n"
            f"问题: {query}"
        ).strip()
    except Exception:
        return None
    if not keywords or len(keywords) > 200:
        return None
    return keywords


# ---------------------------------------------------------------------------
# API server
# ---------------------------------------------------------------------------

_ServerPipeline: RagPipeline | None = None
_query_lock = threading.Lock()


def run_server(args: argparse.Namespace) -> int:
    """Start a local HTTP server that keeps the RAG pipeline in memory."""
    global _ServerPipeline

    store_dir = args.store_dir.resolve()
    if not (store_dir / "manifest.json").exists():
        raise SystemExit(f"No vector DB at {store_dir}. Run build first.")

    dimensions = _stored_dimensions(store_dir)
    model = _embedding_model(args.embedding, dimensions)
    answer_model = _answer_model(args.answer_model)
    pipeline = _pipeline_for(store_dir, model, answer_model=answer_model, args=args)

    if not getattr(args, "dense_only", False):
        print("Warming up BM25 index...", flush=True)
        pipeline._retrieve(query="warmup", k=1, filter=None)

    print(f"Ready. {len(pipeline._chunk_store)} chunks indexed.", flush=True)
    _ServerPipeline = pipeline

    try:
        import uvicorn
    except ImportError:
        raise SystemExit("Install uvicorn: pip install uvicorn")
    if app is None:
        raise SystemExit("Install fastapi: pip install fastapi")

    print(f"Server starting at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


try:
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="AgenticKB RAG Server")

    class QueryRequest(BaseModel):
        question: str
        top_k: int = 20

    class QueryResponse(BaseModel):
        answer: str
        chunks: list[dict]

    @app.get("/")
    def ui():
        from fastapi.responses import HTMLResponse
        ui_path = _PROJECT / "scripts" / "ui.html"
        return HTMLResponse(ui_path.read_text(encoding="utf-8"))

    @app.get("/health")
    def health():
        n = len(_ServerPipeline._chunk_store) if _ServerPipeline else 0
        return {"status": "ok", "chunks_indexed": n}

    @app.post("/query")
    def query_endpoint(req: QueryRequest) -> QueryResponse:
        if _ServerPipeline is None:
            return QueryResponse(answer="Server not ready.", chunks=[])
        with _query_lock:
            result = _ServerPipeline.answer(req.question, k=req.top_k)
        chunks = [
            {
                "chunk_id": hit.chunk.id,
                "document_id": hit.chunk.document_id,
                "text": hit.chunk.text,
                "score": hit.score,
                "source_uri": unquote(hit.chunk.metadata.attributes.get("source_uri", "unknown source")),
            }
            for hit in result.context.selected
        ]
        return QueryResponse(answer=result.answer.text, chunks=chunks)

except ImportError:
    app = None


if __name__ == "__main__":
    sys.exit(main())
