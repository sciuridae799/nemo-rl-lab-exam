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


def _corpus_gold_presence(
    expected_values: list[str],
    index: LocalMarkdownIndex,
) -> set[str]:
    """单次扫描语料，找出离线 gold 要点中确实存在的规范化文本。"""
    alternatives = {
        alternative
        for expected in expected_values
        for item in _gold_items(expected)[1]
        for alternative in item
        if len(alternative) >= 2 or alternative.isdigit()
    }
    numeric = {alternative for alternative in alternatives if alternative.isdigit()}
    textual = alternatives - numeric
    found: set[str] = set()
    matcher = None
    if textual:
        choices = "|".join(
            re.escape(alternative)
            for alternative in sorted(textual, key=lambda value: (-len(value), value))
        )
        # 正向预查允许有重叠的 gold 文本在同一位置分别被发现。
        matcher = re.compile(f"(?=({choices}))")

    for chunk in index.chunks:
        passage = chunk.source + "\n" + chunk.text
        if matcher is not None:
            compact_passage = compact_text(passage)
            found.update(match.group(1) for match in matcher.finditer(compact_passage))
        if numeric:
            ascii_tokens = set(
                re.findall(r"[a-z0-9][a-z0-9_.+-]*", passage.lower())
            )
            found.update(numeric & ascii_tokens)

    # 若语料包含一个更长的答案串，它也必然覆盖其内部的短 gold 串。
    matched_text = tuple(found)
    found.update(
        alternative
        for alternative in textual
        if any(alternative in match for match in matched_text)
    )
    return found


def _presence_coverage(expected: str, present: set[str]) -> float:
    _, items = _gold_items(expected)
    if not items:
        return 0.0
    return sum(any(alternative in present for alternative in item) for item in items) / len(items)


def _mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


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
    candidate_max_per_source: int = 4,
    query_expansion: bool = False,
    structural_expansion: bool = False,
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

    corpus_presence = _corpus_gold_presence(
        [str(row[output_key]) for _, row, _ in selected],
        index,
    )

    baseline_rows: list[dict[str, float]] = []
    reranked_rows: list[dict[str, float]] = []
    per_type: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: {"baseline": [], "reranked": []})
    funnel_values: dict[str, list[float]] = defaultdict(list)
    funnel_by_type: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    candidate_hit_counts: list[float] = []
    expanded_hit_counts: list[float] = []
    structural_changed = 0
    structural_coverage_improved = 0
    candidate_missed_corpus = 0
    reranker_dropped_evidence = 0
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
            max_per_source=candidate_max_per_source,
        )
        raw_candidate_hits = list(candidate_hits)
        if structural_expansion:
            candidate_hits = index.expand_structural_candidates(
                question,
                candidate_hits,
            )
        reranked_hits = rerank_answerable_hits(
            question,
            candidate_hits,
            top_k=top_k,
            baseline_hits=baseline_hits[:1],
        )
        baseline = _retrieval_stats(question, expected, baseline_hits)
        reranked = _retrieval_stats(question, expected, reranked_hits)
        corpus_coverage = _presence_coverage(expected, corpus_presence)
        candidate_coverage = evidence_coverage(expected, raw_candidate_hits)
        expanded_coverage = evidence_coverage(expected, candidate_hits)
        baseline_rows.append(baseline)
        reranked_rows.append(reranked)
        per_type[question_type]["baseline"].append(baseline)
        per_type[question_type]["reranked"].append(reranked)

        stage_coverages = {
            "corpus": corpus_coverage,
            "baseline_top3": baseline["evidence_coverage"],
            "candidate_pool": candidate_coverage,
            "expanded_pool": expanded_coverage,
            "reranked_top3": reranked["evidence_coverage"],
        }
        for stage, coverage in stage_coverages.items():
            funnel_values[stage].append(coverage)
            funnel_by_type[question_type][stage].append(coverage)
        candidate_hit_counts.append(float(len(raw_candidate_hits)))
        expanded_hit_counts.append(float(len(candidate_hits)))
        raw_keys = {(hit.source, hit.text) for hit in raw_candidate_hits}
        expanded_keys = {(hit.source, hit.text) for hit in candidate_hits}
        structural_changed += int(expanded_keys != raw_keys)
        structural_coverage_improved += int(expanded_coverage > candidate_coverage)
        candidate_missed_corpus += int(corpus_coverage > candidate_coverage)
        reranker_dropped_evidence += int(expanded_coverage > reranked["evidence_coverage"])

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
    funnel_summary = {
        f"{stage}_evidence_coverage": _mean(values)
        for stage, values in sorted(funnel_values.items())
    }
    funnel_summary.update({
        "candidate_recall_gap": (
            funnel_summary["corpus_evidence_coverage"]
            - funnel_summary["candidate_pool_evidence_coverage"]
        ),
        "structural_gain": (
            funnel_summary["expanded_pool_evidence_coverage"]
            - funnel_summary["candidate_pool_evidence_coverage"]
        ),
        "top3_selection_gap": (
            funnel_summary["expanded_pool_evidence_coverage"]
            - funnel_summary["reranked_top3_evidence_coverage"]
        ),
        "mean_candidate_hits": _mean(candidate_hit_counts),
        "mean_expanded_hits": _mean(expanded_hit_counts),
        "structural_changed_samples": structural_changed,
        "structural_coverage_improved_samples": structural_coverage_improved,
        "candidate_missed_corpus_samples": candidate_missed_corpus,
        "reranker_dropped_evidence_samples": reranker_dropped_evidence,
    })
    return {
        "sample_count": len(selected),
        "max_per_type": int(max_per_type),
        "seed": int(seed),
        "top_k": int(top_k),
        "candidate_k": int(candidate_k),
        "candidate_max_per_source": int(candidate_max_per_source),
        "query_expansion": bool(query_expansion),
        "structural_expansion": bool(structural_expansion),
        "funnel": funnel_summary,
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
                "funnel": {
                    f"{stage}_evidence_coverage": _mean(stage_values)
                    for stage, stage_values in sorted(
                        funnel_by_type[question_type].items()
                    )
                },
            }
            for question_type, values in sorted(per_type.items())
        },
        "changed_examples": examples,
    }
