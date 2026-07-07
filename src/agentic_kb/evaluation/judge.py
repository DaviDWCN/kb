"""LLM-as-judge evaluation for RAG answers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


class JudgeOutputError(ValueError):
    """Raised when a judge model returns malformed or unusable JSON."""


@dataclass(frozen=True)
class JudgeCriterion:
    """One scored dimension used by the LLM judge."""

    name: str
    description: str
    weight: float = 1.0


@dataclass(frozen=True)
class JudgeContext:
    """Context chunk shown to the judge as evidence."""

    chunk_id: str
    text: str
    source_uri: str | None = None


@dataclass(frozen=True)
class JudgeCase:
    """One answer-quality case to evaluate."""

    id: str
    question: str
    answer: str
    context: tuple[JudgeContext, ...] = ()
    reference_answer: str | None = None
    expected_facts: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()


@dataclass(frozen=True)
class JudgeResult:
    """Structured judgment for one case."""

    case_id: str
    scores: dict[str, float]
    overall_score: float
    passed: bool
    explanation: str
    issues: tuple[str, ...] = ()
    raw_response: str = ""


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate view across many judge results."""

    results: list[JudgeResult]
    case_count: int
    passed_count: int
    pass_rate: float
    average_score: float
    average_scores: dict[str, float] = field(default_factory=dict)


DEFAULT_JUDGE_CRITERIA = (
    JudgeCriterion(
        "correctness",
        "The answer directly and accurately answers the question. Use the reference answer "
        "and expected facts when they are provided.",
    ),
    JudgeCriterion(
        "faithfulness",
        "The answer is supported by the provided context and does not invent document facts.",
    ),
    JudgeCriterion(
        "completeness",
        "The answer includes the important information needed to satisfy the question.",
    ),
    JudgeCriterion(
        "citation_quality",
        "The citations or cited chunks support the answer and point to relevant evidence.",
    ),
)


class LlmJudge:
    """Evaluate RAG answers by asking an LLM for strict JSON scores."""

    def __init__(
        self,
        model,
        *,
        criteria: tuple[JudgeCriterion, ...] = DEFAULT_JUDGE_CRITERIA,
        pass_threshold: float = 0.75,
    ) -> None:
        self._model = model
        self._criteria = _validate_criteria(criteria)
        if not 0 <= pass_threshold <= 1:
            raise ValueError("pass_threshold must be between 0 and 1")
        self._pass_threshold = pass_threshold

    def evaluate(self, case: JudgeCase) -> JudgeResult:
        """Judge one case and return validated scores."""
        _validate_case(case)
        raw_response = self._model.generate(_prompt_for(case, self._criteria)).strip()
        payload = _parse_json_object(raw_response)
        scores = _scores_from_payload(payload, self._criteria)
        overall = _weighted_average(scores, self._criteria)

        return JudgeResult(
            case_id=case.id,
            scores=scores,
            overall_score=overall,
            passed=overall >= self._pass_threshold,
            explanation=_string_field(payload, "explanation"),
            issues=_string_tuple_field(payload, "issues"),
            raw_response=raw_response,
        )

    def evaluate_cases(self, cases: list[JudgeCase]) -> EvaluationSummary:
        """Judge many cases and return aggregate metrics."""
        results = [self.evaluate(case) for case in cases]
        return _summary_for(results, self._criteria)


def evaluate_with_llm_judge(
    model,
    cases: list[JudgeCase],
    *,
    criteria: tuple[JudgeCriterion, ...] = DEFAULT_JUDGE_CRITERIA,
    pass_threshold: float = 0.75,
) -> EvaluationSummary:
    """Convenience function for one-off LLM-as-judge evaluation runs."""
    return LlmJudge(model, criteria=criteria, pass_threshold=pass_threshold).evaluate_cases(cases)


def _prompt_for(case: JudgeCase, criteria: tuple[JudgeCriterion, ...]) -> str:
    criteria_lines = "\n".join(
        f"- {criterion.name}: {criterion.description} Score from 0.0 to 1.0."
        for criterion in criteria
    )
    response_template = json.dumps(
        {
            "scores": {criterion.name: 0.0 for criterion in criteria},
            "explanation": "brief reason",
            "issues": ["optional issue"],
        },
        indent=2,
    )
    context_blocks = "\n\n".join(_context_block(context) for context in case.context)
    if not context_blocks:
        context_blocks = "No retrieved context was provided."

    expected_facts = "\n".join(f"- {fact}" for fact in case.expected_facts) or "None provided."
    citations = "\n".join(f"- {citation}" for citation in case.citations) or "None provided."
    reference_answer = case.reference_answer or "None provided."

    return (
        "You are judging the quality of a RAG answer.\n"
        "Use only the question, reference answer, expected facts, retrieved context, "
        "and citations shown below. Do not reward unsupported claims.\n\n"
        "Return only valid JSON with this shape:\n"
        f"{response_template}\n\n"
        "Scoring Criteria:\n"
        f"{criteria_lines}\n\n"
        f"Question:\n{case.question.strip()}\n\n"
        f"Answer:\n{case.answer.strip()}\n\n"
        f"Reference Answer:\n{reference_answer}\n\n"
        f"Expected Facts:\n{expected_facts}\n\n"
        f"Citations Used:\n{citations}\n\n"
        f"Retrieved Context:\n{context_blocks}"
    )


def _context_block(context: JudgeContext) -> str:
    source = f" source={context.source_uri}" if context.source_uri else ""
    return f"[{context.chunk_id}{source}]\n{context.text.strip()}"


def _repair_json(text: str) -> Any:
    """Attempt to salvage scores from malformed LLM JSON."""
    print("[REPAIR] Using NEW repair function")
    import re

    cleaned = text.strip()
    # Strip markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Extract scores: match "key": NUMBER or "key": junk_score → find any float nearby
    scores: dict[str, float] = {}
    score_keys = ["correctness", "faithfulness", "completeness", "citation_quality"]
    for m in re.finditer(r'"(correctness|faithfulness|completeness|citation_quality)":\s*', cleaned):
        key = m.group(1)
        after = cleaned[m.end():]
        num_match = re.match(r'\s*(\d+\.?\d*)', after)
        if num_match:
            scores[key] = float(num_match.group(1))
    # Default any missing score keys to 0.0
    for k in score_keys:
        scores.setdefault(k, 0.0)
    print(f"[REPAIR] Extracted scores: {scores}")

    # Extract explanation  
    expl_match = re.search(r'"explanation":\s*"([^"]*)"', cleaned)
    explanation = expl_match.group(1) if expl_match else ""

    # Extract issues array: find [ ... ] after "issues"
    issues: list[str] = []
    issues_start = cleaned.find('"issues"')
    if issues_start >= 0:
        bracket = cleaned.find('[', issues_start)
        close = cleaned.find(']', bracket + 1) if bracket >= 0 else -1
        if bracket >= 0 and close > bracket:
            issues_text = cleaned[bracket + 1:close]
            issues = [i.strip().strip('"\'') for i in issues_text.split('","') if i.strip().strip('"\'')]

    return {"scores": scores, "explanation": explanation, "issues": issues}


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = _parse_embedded_json(text)
    if not isinstance(payload, dict):
        raise JudgeOutputError("judge response must be a JSON object")
    return payload


def _parse_embedded_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
        try:
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise JudgeOutputError("judge response did not contain a JSON object")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        try:
            payload = _repair_json(text[start : end + 1])
        except Exception as repair_error:
            raise JudgeOutputError(f"judge response was not valid JSON: {repair_error}") from repair_error
    if not isinstance(payload, dict):
        raise JudgeOutputError("judge response must be a JSON object")
    return payload


def _scores_from_payload(
    payload: dict[str, Any],
    criteria: tuple[JudgeCriterion, ...],
) -> dict[str, float]:
    raw_scores = payload.get("scores")
    if not isinstance(raw_scores, dict):
        raise JudgeOutputError("judge response must include a scores object")

    scores: dict[str, float] = {}
    for criterion in criteria:
        raw_score = raw_scores.get(criterion.name)
        if not isinstance(raw_score, int | float):
            raise JudgeOutputError(f"score for {criterion.name} must be a number")
        score = float(raw_score)
        if not 0 <= score <= 1:
            raise JudgeOutputError(f"score for {criterion.name} must be between 0 and 1")
        scores[criterion.name] = score
    return scores


def _weighted_average(scores: dict[str, float], criteria: tuple[JudgeCriterion, ...]) -> float:
    total_weight = sum(criterion.weight for criterion in criteria)
    return sum(scores[criterion.name] * criterion.weight for criterion in criteria) / total_weight


def _summary_for(
    results: list[JudgeResult],
    criteria: tuple[JudgeCriterion, ...],
) -> EvaluationSummary:
    if not results:
        return EvaluationSummary(
            results=[],
            case_count=0,
            passed_count=0,
            pass_rate=0.0,
            average_score=0.0,
            average_scores={criterion.name: 0.0 for criterion in criteria},
        )

    passed_count = sum(1 for result in results if result.passed)
    return EvaluationSummary(
        results=results,
        case_count=len(results),
        passed_count=passed_count,
        pass_rate=passed_count / len(results),
        average_score=sum(result.overall_score for result in results) / len(results),
        average_scores={
            criterion.name: sum(result.scores[criterion.name] for result in results) / len(results)
            for criterion in criteria
        },
    )


def _validate_criteria(criteria: tuple[JudgeCriterion, ...]) -> tuple[JudgeCriterion, ...]:
    if not criteria:
        raise ValueError("criteria must not be empty")

    seen: set[str] = set()
    for criterion in criteria:
        if not criterion.name.strip():
            raise ValueError("criterion names must not be empty")
        if criterion.name in seen:
            raise ValueError(f"duplicate criterion name: {criterion.name}")
        if criterion.weight <= 0:
            raise ValueError("criterion weights must be greater than zero")
        seen.add(criterion.name)
    return criteria


def _validate_case(case: JudgeCase) -> None:
    if not case.id.strip():
        raise ValueError("case id must not be empty")
    if not case.question.strip():
        raise ValueError("question must not be empty")
    if not case.answer.strip():
        raise ValueError("answer must not be empty")


def _string_field(payload: dict[str, Any], name: str) -> str:
    value = payload.get(name, "")
    return value if isinstance(value, str) else ""


def _string_tuple_field(payload: dict[str, Any], name: str) -> tuple[str, ...]:
    value = payload.get(name, [])
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))
