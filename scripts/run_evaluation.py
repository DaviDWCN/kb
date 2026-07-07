#!/usr/bin/env python3
"""Run LLM-as-judge evaluation from a JSONL cases file."""

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
from typing import Any

from agentic_kb.config import load_env_file
from agentic_kb.evaluation import (
    EvaluationSummary,
    JudgeCase,
    JudgeContext,
    RetrievalCase,
    RetrievalSummary,
    evaluate_retrieval,
    evaluate_with_llm_judge,
)
from agentic_kb.providers import DeepSeek, Qwen


def main(argv: list[str] | None = None) -> int:
    load_env_file(_PROJECT / ".env")
    args = _arg_parser().parse_args(argv)
    if args.mode == "retrieval":
        summary = evaluate_retrieval(load_retrieval_cases(args.cases), k=args.k)
        payload = retrieval_summary_to_dict(summary)
        _print_retrieval_summary(summary)
        _write_output(args.output, payload)
        return 0

    cases = load_cases(args.cases)
    summary = evaluate_with_llm_judge(
        judge_model(args.judge_model),
        cases,
        pass_threshold=args.pass_threshold,
    )
    payload = summary_to_dict(summary)
    _print_summary(summary)
    _write_output(args.output, payload)
    return 0


def load_cases(path: Path) -> list[JudgeCase]:
    """Load judge cases from a JSONL file."""
    cases: list[JudgeCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        cases.append(_case_from_payload(payload, path=path, line_number=line_number))
    return cases


def load_retrieval_cases(path: Path) -> list[RetrievalCase]:
    """Load ranked retrieval cases from a JSONL file."""
    cases: list[RetrievalCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        cases.append(_retrieval_case_from_payload(payload, path=path, line_number=line_number))
    return cases


def judge_model(name: str):
    """Build the LLM adapter used as the judge."""
    cleaned_name = name.strip().lower()
    if cleaned_name == "qwen":
        return Qwen()
    if cleaned_name == "deepseek":
        return DeepSeek()
    raise ValueError(f"unsupported judge model: {name}")


def summary_to_dict(summary: EvaluationSummary) -> dict[str, Any]:
    """Convert summary dataclasses into JSON-serializable dictionaries."""
    return {
        "case_count": summary.case_count,
        "passed_count": summary.passed_count,
        "pass_rate": summary.pass_rate,
        "average_score": summary.average_score,
        "average_scores": dict(summary.average_scores),
        "results": [
            {
                "case_id": result.case_id,
                "overall_score": result.overall_score,
                "passed": result.passed,
                "scores": dict(result.scores),
                "explanation": result.explanation,
                "issues": list(result.issues),
                "raw_response": result.raw_response,
            }
            for result in summary.results
        ],
    }


def retrieval_summary_to_dict(summary: RetrievalSummary) -> dict[str, Any]:
    """Convert retrieval metrics into a JSON-serializable dictionary."""
    return {
        "k": summary.k,
        "case_count": summary.case_count,
        "hit_rate_at_k": summary.hit_rate_at_k,
        "average_precision_at_k": summary.average_precision_at_k,
        "average_recall_at_k": summary.average_recall_at_k,
        "average_mrr_at_k": summary.average_mrr_at_k,
        "average_ndcg_at_k": summary.average_ndcg_at_k,
        "results": [
            {
                "case_id": result.case_id,
                "k": result.k,
                "hit_at_k": result.hit_at_k,
                "precision_at_k": result.precision_at_k,
                "recall_at_k": result.recall_at_k,
                "mrr_at_k": result.mrr_at_k,
                "ndcg_at_k": result.ndcg_at_k,
                "relevant_retrieved_count": result.relevant_retrieved_count,
                "relevant_count": result.relevant_count,
                "retrieved_count": result.retrieved_count,
            }
            for result in summary.results
        ],
    }


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RAG evaluation.")
    parser.add_argument("cases", type=Path, help="JSONL file of evaluation cases.")
    parser.add_argument("--mode", choices=("judge", "retrieval"), default="judge")
    parser.add_argument("--judge-model", choices=("qwen", "deepseek"), default="qwen")
    parser.add_argument("--pass-threshold", type=float, default=0.75)
    parser.add_argument("--k", type=int, default=5, help="Retrieval cutoff for retrieval mode.")
    parser.add_argument("--output", type=Path, help="Optional path to write JSON results.")
    return parser


def _case_from_payload(payload: dict[str, Any], *, path: Path, line_number: int) -> JudgeCase:
    return JudgeCase(
        id=_required_string(payload, "id", path, line_number),
        question=_required_string(payload, "question", path, line_number),
        answer=_required_string(payload, "answer", path, line_number),
        context=tuple(_context_from_payload(item, path, line_number) for item in _list(payload, "context")),
        reference_answer=_optional_string(payload, "reference_answer"),
        expected_facts=_string_tuple(payload, "expected_facts"),
        citations=_string_tuple(payload, "citations"),
    )


def _retrieval_case_from_payload(
    payload: dict[str, Any],
    *,
    path: Path,
    line_number: int,
) -> RetrievalCase:
    return RetrievalCase(
        id=_required_string(payload, "id", path, line_number),
        query=_required_string(payload, "query", path, line_number),
        relevant_chunk_ids=_required_string_tuple(
            payload,
            "relevant_chunk_ids",
            path,
            line_number,
        ),
        retrieved_chunk_ids=_required_string_tuple(
            payload,
            "retrieved_chunk_ids",
            path,
            line_number,
        ),
    )


def _context_from_payload(payload: Any, path: Path, line_number: int) -> JudgeContext:
    if not isinstance(payload, dict):
        raise ValueError(f"{path}:{line_number} context entries must be JSON objects")
    return JudgeContext(
        chunk_id=_required_string(payload, "chunk_id", path, line_number),
        text=_required_string(payload, "text", path, line_number),
        source_uri=_optional_string(payload, "source_uri"),
    )


def _required_string(
    payload: dict[str, Any],
    key: str,
    path: Path,
    line_number: int,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}:{line_number} field {key!r} must be a non-empty string")
    return value


def _optional_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) and value.strip() else None


def _list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key, [])
    return value if isinstance(value, list) else []


def _string_tuple(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    return tuple(item for item in _list(payload, key) if isinstance(item, str))


def _required_string_tuple(
    payload: dict[str, Any],
    key: str,
    path: Path,
    line_number: int,
) -> tuple[str, ...]:
    values = _string_tuple(payload, key)
    if not values:
        raise ValueError(f"{path}:{line_number} field {key!r} must contain strings")
    return values


def _write_output(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _print_summary(summary: EvaluationSummary) -> None:
    print("LLM Judge Evaluation")
    print("--------------------")
    print(f"Cases: {summary.case_count}")
    print(f"Passed: {summary.passed_count}")
    print(f"Pass rate: {summary.pass_rate:.2%}")
    print(f"Average score: {summary.average_score:.3f}")
    for result in summary.results:
        status = "PASS" if result.passed else "FAIL"
        print(f"- {result.case_id}: {status} score={result.overall_score:.3f}")


def _print_retrieval_summary(summary: RetrievalSummary) -> None:
    print("Retrieval Evaluation")
    print("--------------------")
    print(f"Cases: {summary.case_count}")
    print(f"K: {summary.k}")
    print(f"Hit rate@k: {summary.hit_rate_at_k:.2%}")
    print(f"Precision@k: {summary.average_precision_at_k:.3f}")
    print(f"Recall@k: {summary.average_recall_at_k:.3f}")
    print(f"MRR@k: {summary.average_mrr_at_k:.3f}")
    print(f"NDCG@k: {summary.average_ndcg_at_k:.3f}")
    for result in summary.results:
        status = "HIT" if result.hit_at_k else "MISS"
        print(f"- {result.case_id}: {status} recall={result.recall_at_k:.3f}")


if __name__ == "__main__":
    sys.exit(main())
