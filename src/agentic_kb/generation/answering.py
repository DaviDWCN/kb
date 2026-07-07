"""Grounded answer generation over selected context."""

from dataclasses import dataclass

from agentic_kb.retrieval import CitationBuilder, CitationSource, ContextSelection


NO_CONTEXT_PREFIX = "未在提供的文档中找到相关信息。"


@dataclass(frozen=True)
class Answer:
    """Generated answer and the context sources used to produce it."""

    query: str
    text: str
    citations: list[CitationSource]
    used_chunk_ids: list[str]


class AnswerGenerator:
    """Generate an answer from selected retrieval context."""

    def __init__(self, model, citation_builder: CitationBuilder) -> None:
        self._model = model
        self._citation_builder = citation_builder

    def answer(self, query: str, context: ContextSelection) -> Answer:
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("query must not be empty")

        prompt = _prompt_for(cleaned_query, context)
        text = self._model.generate(prompt).strip()
        if not context.selected:
            text = _ensure_no_context_prefix(text)

        citations = self._citation_builder.build(context.selected)
        return Answer(
            query=cleaned_query,
            text=text,
            citations=citations,
            used_chunk_ids=[hit.chunk.id for hit in context.selected],
        )


def _prompt_for(query: str, context: ContextSelection) -> str:
    if not context.selected:
        return _no_context_prompt(query)

    context_blocks = "\n\n".join(
        _format_chunk(hit) for hit in context.selected
    )
    return (
        "你是一个保险行业的知识助手。请根据以下文档上下文，详细、完整地回答用户问题。\n"
        "如果有多个要点，用编号列出。如果涉及不同系统或项目，分别说明。\n"
        "不要只说结论，要解释原因和依据。全程使用中文。\n"
        "如果上下文不足以回答问题，说明上下文实际包含了什么、缺了什么。不要编造。\n"
        f"用户问题:\n{query}\n\n"
        f"文档上下文:\n{context_blocks}\n\n"
        "回答:"
    )


def _format_chunk(hit) -> str:
    chunk = hit.chunk
    source = chunk.metadata.attributes.get("source_uri", chunk.document_id)
    page = f"页码: {chunk.metadata.page_number}" if chunk.metadata.page_number is not None else ""
    mtime = chunk.metadata.attributes.get("file_mtime", "")
    header = source
    if page:
        header += f" | {page}"
    if mtime:
        header += f" | 修改日期: {mtime}"
    return f"{header}\n{chunk.text.strip()}"


def _no_context_prompt(query: str) -> str:
    return (
        "文档库中没有找到与问题相关的内容。\n"
        f"问题:\n{query}\n\n"
        "请严格按照以下要求回答：\n"
        f"1. 第一句必须以「{NO_CONTEXT_PREFIX}」开头。\n"
        "2. 然后用中文给出通用建议，明确说明这些建议不来自提供的文档。\n"
        "3. 列出可能的原因和假设。\n"
        "4. 全程使用中文回答。\n\n"
        "回答:"
    )


def _ensure_no_context_prefix(text: str) -> str:
    if text.startswith(NO_CONTEXT_PREFIX):
        return text
    if not text:
        return NO_CONTEXT_PREFIX
    return f"{NO_CONTEXT_PREFIX} {text}"
