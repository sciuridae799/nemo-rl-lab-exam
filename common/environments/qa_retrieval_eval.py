"""在不训练模型的情况下比较 QA 文档检索方案。"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from statistics import fmean
from typing import Any

from common.environments.qa_search_core import (
    LocalMarkdownIndex,
    SearchHit,
    _question_text,
    build_query_variants,
    compact_text,
    question_copy_score,
    rerank_answerable_hits,
)

_EXPECTED = re.compile(r"\s*\[(\w+)]\s*(.*)", re.DOTALL)
_OPEN_TYPES = {"fill", "short"}


def _gold_items(expected: str) -> tuple[str, list[list[str]]]:
    match = _EXPECTED.match(str(expected))
    if not match:
        return "unknown", []
    question_type, gold = match.group(1), match.group(2)
    items: list[list[str]] = []
    for item in gold.split("|||"):
        alternatives = [compact_text(part) for part in re.split(r"[/／]", item) if compact_text(part)]
        if alternatives:
            items.append(alternatives)
    return question_type, items


def evidence_coverage(expected: str, hits: list[SearchHit]) -> float:
    """计算 gold 要点在返回证据中的覆盖率，仅用于离线评估。"""
    _, items = _gold_items(expected)
    if not items:
        return 0.0
    passage = "\n".join(hit.source + "\n" + hit.text for hit in hits)
    compact_passage = compact_text(passage)
    ascii_tokens = set(re.findall(r"[a-z0-9][a-z0-9_.+-]*", passage.lower()))
    covered = 0
    for alternatives in items:
        matched = any(
            (len(alternative) >= 2 and alternative in compact_passage)
            or (alternative.isdigit() and alternative in ascii_tokens)
            for alternative in alternatives
        )
        covered += int(matched)
    return covered / len(items)


def _retrieval_stats(question: str, expected: str, hits: list[SearchHit]) -> dict[str, float]:
    return {
        "evidence_coverage": evidence_coverage(expected, hits),
        "question_copy_rate": (
            sum(question_copy_score(question, hit.text) >= 0.5 for hit in hits) / len(hits) if hits else 0.0
        ),
        "result_chars": float(sum(len(hit.text) for hit in hits)),
    }


def _mean_stats(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = ("evidence_coverage", "question_copy_rate", "result_chars")
    return {key: fmean(row[key] for row in rows) if rows else 0.0 for key in keys}


def evaluate_retrieval_ab(
    rows: list[dict[str, Any]],
    index: LocalMarkdownIndex,
    *,
    input_key: str = "query",
    output_key: str = "expected_answer",
    max_per_type: int = 64,
    seed: int = 42,
    top_k: int = 3,
    candidate_k: int = 20,
    query_expansion: bool = False,
) -> dict[str, Any]:
    """在训练集开放题上比较原始 BM25 与可回答性重排。"""
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        question_type, _ = _gold_items(str(row.get(output_key, "")))
        if question_type in _OPEN_TYPES:
            grouped[question_type].append((row_index, row))

    rng = random.Random(seed)
    selected: list[tuple[int, dict[str, Any], str]] = []
    for question_type in sorted(_OPEN_TYPES):
        candidates = list(grouped.get(question_type, []))
        rng.shuffle(candidates)
        selected.extend((row_index, row, question_type) for row_index, row in candidates[: max(1, int(max_per_type))])

    baseline_rows: list[dict[str, float]] = []
    reranked_rows: list[dict[str, float]] = []
    per_type: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: {"baseline": [], "reranked": []})
    examples: list[dict[str, Any]] = []
    improved = 0
    regressed = 0

    for row_index, row, question_type in selected:
        query = str(row[input_key])
        expected = str(row[output_key])
        question = _question_text(query)
        baseline_hits = index.search(question, top_k=top_k)
        candidate_queries = (
            build_query_variants(question)
            if query_expansion
            else [question]
        )
        candidate_hits = index.search_union(
            candidate_queries,
            candidate_k=candidate_k,
        )
        reranked_hits = rerank_answerable_hits(
            question,
            candidate_hits,
            top_k=top_k,
            baseline_hits=baseline_hits[:1],
        )
        baseline = _retrieval_stats(question, expected, baseline_hits)
        reranked = _retrieval_stats(question, expected, reranked_hits)
        baseline_rows.append(baseline)
        reranked_rows.append(reranked)
        per_type[question_type]["baseline"].append(baseline)
        per_type[question_type]["reranked"].append(reranked)

        delta = reranked["evidence_coverage"] - baseline["evidence_coverage"]
        improved += int(delta > 0)
        regressed += int(delta < 0)
        if delta != 0 and len(examples) < 12:
            examples.append(
                {
                    "row_index": row_index,
                    "type": question_type,
                    "question": question[:180],
                    "baseline_coverage": baseline["evidence_coverage"],
                    "reranked_coverage": reranked["evidence_coverage"],
                    "baseline_sources": [hit.source for hit in baseline_hits],
                    "reranked_sources": [hit.source for hit in reranked_hits],
                }
            )

    baseline_summary = _mean_stats(baseline_rows)
    reranked_summary = _mean_stats(reranked_rows)
    return {
        "sample_count": len(selected),
        "max_per_type": int(max_per_type),
        "seed": int(seed),
        "top_k": int(top_k),
        "candidate_k": int(candidate_k),
        "query_expansion": bool(query_expansion),
        "baseline": baseline_summary,
        "reranked": reranked_summary,
        "delta": {key: reranked_summary[key] - baseline_summary[key] for key in baseline_summary},
        "improved_samples": improved,
        "regressed_samples": regressed,
        "by_type": {
            question_type: {
                "count": len(values["baseline"]),
                "baseline": _mean_stats(values["baseline"]),
                "reranked": _mean_stats(values["reranked"]),
            }
            for question_type, values in sorted(per_type.items())
        },
        "changed_examples": examples,
    }
