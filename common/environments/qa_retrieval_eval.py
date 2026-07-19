"""在不训练模型的情况下比较 QA 文档检索方案。"""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from statistics import fmean
from typing import Any

import numpy as np

from common.environments.qa_search_core import (
    ANSWERABILITY_FEATURE_NAMES,
    LocalMarkdownIndex,
    SearchHit,
    _question_text,
    answerability_feature_rows,
    build_query_variants,
    compact_text,
    extract_answerable_snippets,
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


def _supervised_answer_text(expected: str, *, max_chars: int = 180) -> str:
    """仅用于训练集检索扩展的答案词串，不向模型暴露 gold。

    只保留填空/简答的核心字段，限长并去掉题型标记，以免将整段标准答案作为查询。
    """
    question_type, items = _gold_items(expected)
    if question_type not in _OPEN_TYPES:
        return ""
    values: list[str] = []
    for alternatives in items:
        if alternatives:
            values.append(alternatives[0][:48])
    return " ".join(values)[: max(24, int(max_chars))]


def _stable_holdout(question: str, *, denominator: int = 5) -> bool:
    digest = hashlib.sha1(compact_text(question).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % max(2, int(denominator)) == 0


def evaluate_supervised_query_expansion(
    rows: list[dict[str, Any]],
    index: LocalMarkdownIndex,
    *,
    input_key: str = "query",
    output_key: str = "expected_answer",
    top_k: int = 3,
    candidate_k: int = 80,
    candidate_max_per_source: int = 4,
    query_expansion: bool = True,
    structural_expansion: bool = True,
    aligned_sibling_expansion: bool = True,
    max_neighbors: int = 3,
    min_similarity: float = 0.25,
) -> dict[str, Any]:
    """用训练题目的重新编码进行查询扩展，按规范化题干做 held-out 门控。

    训练折只用训练题的文本和期望答案拟合 TF-IDF 近邻；测试折的期望答案仅用于离线评估。
    这个函数不读验证集，可在正式环境中将拟合器只用训练数据构建。
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:  # pragma: no cover - 远端能力依赖
        raise RuntimeError("supervised query expansion 需要 scikit-learn") from exc

    open_rows = [
        row
        for row in rows
        if _gold_items(str(row.get(output_key, "")))[0] in _OPEN_TYPES
    ]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in open_rows:
        groups[compact_text(_question_text(str(row.get(input_key, ""))))].append(row)

    train_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    for group_rows in groups.values():
        if _stable_holdout(str(group_rows[0].get(input_key, ""))):
            holdout_rows.extend(group_rows)
        else:
            train_rows.extend(group_rows)
    if not holdout_rows or not train_rows:
        raise ValueError("监督查询扩展分折后 train/holdout 为空")

    train_questions = [
        _question_text(str(row[input_key])) for row in train_rows
    ]
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=1,
        max_features=60_000,
        sublinear_tf=True,
    )
    train_matrix = vectorizer.fit_transform(train_questions)
    train_answers = [
        _supervised_answer_text(str(row[output_key])) for row in train_rows
    ]

    baseline_values: list[dict[str, float]] = []
    supervised_values: list[dict[str, float]] = []
    by_type: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"baseline": [], "supervised": []}
    )
    neighbor_similarities: list[float] = []
    neighbor_counts: list[float] = []
    changed = 0

    for row in holdout_rows:
        question = _question_text(str(row[input_key]))
        expected = str(row[output_key])
        base_queries = build_query_variants(question) if query_expansion else [question]
        base_candidates = index.search_union(
            base_queries,
            candidate_k=candidate_k,
            max_per_source=candidate_max_per_source,
        )
        if structural_expansion:
            base_candidates = index.expand_structural_candidates(
                question,
                base_candidates,
                include_aligned_siblings=aligned_sibling_expansion,
            )
        base_hits = rerank_answerable_hits(
            question,
            base_candidates,
            top_k=top_k,
            baseline_hits=index.search(question, top_k=top_k)[:1],
        )

        query_vector = vectorizer.transform([question])
        similarities = (train_matrix @ query_vector.T).toarray().ravel()
        neighbor_positions = sorted(
            range(len(similarities)),
            key=lambda position: (float(similarities[position]), -position),
            reverse=True,
        )
        neighbor_positions = [
            position
            for position in neighbor_positions[: max(1, int(max_neighbors))]
            if float(similarities[position]) >= float(min_similarity)
            and train_answers[position]
        ]
        neighbor_similarities.extend(float(similarities[position]) for position in neighbor_positions)
        neighbor_counts.append(float(len(neighbor_positions)))
        supervised_queries = list(base_queries)
        for position in neighbor_positions:
            answer = train_answers[position]
            supervised_queries.extend((answer, question + "\n" + answer))
        supervised_candidates = index.search_union(
            supervised_queries,
            candidate_k=candidate_k,
            max_per_source=candidate_max_per_source,
        )
        if structural_expansion:
            supervised_candidates = index.expand_structural_candidates(
                question,
                supervised_candidates,
                include_aligned_siblings=aligned_sibling_expansion,
            )
        supervised_hits = rerank_answerable_hits(
            question,
            supervised_candidates,
            top_k=top_k,
            baseline_hits=index.search(question, top_k=top_k)[:1],
        )

        baseline_stats = _retrieval_stats(question, expected, base_hits)
        supervised_stats = _retrieval_stats(question, expected, supervised_hits)
        baseline_values.append(baseline_stats)
        supervised_values.append(supervised_stats)
        question_type = _gold_items(expected)[0]
        by_type[question_type]["baseline"].append(baseline_stats["evidence_coverage"])
        by_type[question_type]["supervised"].append(supervised_stats["evidence_coverage"])
        changed += int(
            [(hit.source, hit.text) for hit in base_hits]
            != [(hit.source, hit.text) for hit in supervised_hits]
        )

    baseline_summary = _mean_stats(baseline_values)
    supervised_summary = _mean_stats(supervised_values)
    return {
        "sample_count": len(holdout_rows),
        "train_count": len(train_rows),
        "unique_question_count": len(groups),
        "baseline": baseline_summary,
        "supervised": supervised_summary,
        "delta": {
            key: supervised_summary[key] - baseline_summary[key]
            for key in baseline_summary
        },
        "changed_samples": changed,
        "mean_neighbor_similarity": _mean(neighbor_similarities),
        "mean_neighbor_count": _mean(neighbor_counts),
        "by_type": {
            question_type: {
                "count": len(values["baseline"]),
                "baseline_evidence_coverage": _mean(values["baseline"]),
                "supervised_evidence_coverage": _mean(values["supervised"]),
            }
            for question_type, values in sorted(by_type.items())
        },
    }


_ANSWERABILITY_WEIGHT_GRID: dict[str, tuple[float, ...]] = {
    "default": (
        0.35, 0.20, 0.10, 0.20, 0.10, 0.05, 0.75, 0.08, -0.10, -0.80, -0.25
    ),
    # 通用角色识别：对答案/讲义文档加权，同时保留句子与填空桥接信号。
    "answer_role": (
        0.15, 0.10, 0.05, 0.45, 0.20, 0.05, 1.00, 0.65, -0.20, -1.20, -0.30
    ),
    "answer_role_strict": (
        0.10, 0.05, 0.02, 0.60, 0.25, 0.05, 1.20, 1.00, -0.35, -1.50, -0.40
    ),
    # 保守版：只略微提高证据句，用于检查是否是过强答案角色所致退化。
    "answer_role_soft": (
        0.25, 0.15, 0.08, 0.30, 0.15, 0.05, 0.85, 0.30, -0.15, -1.00, -0.25
    ),
}


def evaluate_answerability_weight_grid(
    rows: list[dict[str, Any]],
    index: LocalMarkdownIndex,
    *,
    input_key: str = "query",
    output_key: str = "expected_answer",
    max_per_type: int = 32,
    seed: int = 42,
    top_k: int = 3,
    candidate_k: int = 80,
    candidate_max_per_source: int = 4,
    query_expansion: bool = True,
    structural_expansion: bool = True,
    aligned_sibling_expansion: bool = True,
) -> dict[str, Any]:
    """在同一候选池上比较几组通用可回收的证据重排权重。

    这是离线诊断，不读验证答案；只有某组权重在 held-out 上稳定超过默认方案，才考虑写入线上配置。
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        question_type, _ = _gold_items(str(row.get(output_key, "")))
        if question_type in _OPEN_TYPES:
            grouped[question_type].append(row)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for question_type in sorted(_OPEN_TYPES):
        candidates = list(grouped.get(question_type, []))
        rng.shuffle(candidates)
        selected.extend(candidates[: max(1, int(max_per_type))])

    values: dict[str, list[dict[str, float]]] = {
        name: [] for name in _ANSWERABILITY_WEIGHT_GRID
    }
    baseline_values: list[dict[str, float]] = []
    by_type: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {name: [] for name in _ANSWERABILITY_WEIGHT_GRID}
    )
    for row in selected:
        query = str(row[input_key])
        expected = str(row[output_key])
        question = _question_text(query)
        baseline_hits = index.search(question, top_k=top_k)
        queries = build_query_variants(question) if query_expansion else [question]
        candidates = index.search_union(
            queries,
            candidate_k=candidate_k,
            max_per_source=candidate_max_per_source,
        )
        if structural_expansion:
            candidates = index.expand_structural_candidates(
                question,
                candidates,
                include_aligned_siblings=aligned_sibling_expansion,
            )
        baseline_stats = _retrieval_stats(question, expected, baseline_hits)
        baseline_values.append(baseline_stats)
        question_type = _gold_items(expected)[0]
        for name, weights in _ANSWERABILITY_WEIGHT_GRID.items():
            selected_hits = rerank_answerable_hits(
                question,
                candidates,
                top_k=top_k,
                baseline_hits=baseline_hits[:1],
                weights=weights,
            )
            stats = _retrieval_stats(question, expected, selected_hits)
            values[name].append(stats)
            by_type[question_type][name].append(stats["evidence_coverage"])

    baseline_summary = _mean_stats(baseline_values)
    grid_summary = {
        name: {
            **_mean_stats(stats),
            "delta_vs_baseline": _mean_stats(stats)["evidence_coverage"]
            - baseline_summary["evidence_coverage"],
        }
        for name, stats in values.items()
    }
    return {
        "sample_count": len(selected),
        "baseline": baseline_summary,
        "grid": grid_summary,
        "by_type": {
            question_type: {
                name: {
                    "count": len(type_values[name]),
                    "evidence_coverage": _mean(type_values[name]),
                }
                for name in _ANSWERABILITY_WEIGHT_GRID
            }
            for question_type, type_values in sorted(by_type.items())
        },
    }


def _fit_linear_reranker(
    groups: list[dict[str, Any]],
    *,
    l2: float = 1.0,
) -> dict[str, np.ndarray | float]:
    """按题目等权、题内平衡正负候选，拟合可复现的加权岭回归。"""
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    for group in groups:
        features = np.asarray(group["features"], dtype=np.float64)
        labels = np.asarray(group["labels"], dtype=np.float64)
        if not len(labels):
            continue
        positive = labels > 0.0
        positive_count = int(positive.sum())
        negative_count = len(labels) - positive_count
        weights = np.full(len(labels), 1.0 / len(labels), dtype=np.float64)
        if positive_count and negative_count:
            positive_multiplier = min(25.0, negative_count / positive_count)
            weights[positive] *= positive_multiplier
        feature_parts.append(features)
        label_parts.append(labels)
        weight_parts.append(weights)

    if not feature_parts:
        raise ValueError("线性重排器没有可用候选")
    features = np.concatenate(feature_parts)
    labels = np.concatenate(label_parts)
    sample_weights = np.concatenate(weight_parts)
    sample_weights /= sample_weights.sum()

    mean = np.average(features, axis=0, weights=sample_weights)
    variance = np.average((features - mean) ** 2, axis=0, weights=sample_weights)
    scale = np.sqrt(variance)
    scale[scale < 1.0e-8] = 1.0
    standardized = (features - mean) / scale
    design = np.column_stack([standardized, np.ones(len(standardized))])
    weighted_design = design * np.sqrt(sample_weights)[:, None]
    weighted_labels = labels * np.sqrt(sample_weights)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * max(0.0, float(l2))
    regularizer[-1, -1] = 0.0
    gram = weighted_design.T @ weighted_design + regularizer
    target = weighted_design.T @ weighted_labels
    try:
        fitted = np.linalg.solve(gram, target)
    except np.linalg.LinAlgError:
        fitted = np.linalg.lstsq(gram, target, rcond=None)[0]
    return {
        "mean": mean,
        "scale": scale,
        "weights": fitted[:-1],
        "intercept": float(fitted[-1]),
    }


def _predict_linear_reranker(
    model: dict[str, np.ndarray | float],
    features: np.ndarray,
) -> np.ndarray:
    standardized = (
        np.asarray(features, dtype=np.float64) - np.asarray(model["mean"])
    ) / np.asarray(model["scale"])
    return standardized @ np.asarray(model["weights"]) + float(model["intercept"])


def _select_linear_hits(
    hits: list[SearchHit],
    predictions: np.ndarray,
    *,
    top_k: int,
    max_per_source: int = 2,
) -> list[SearchHit]:
    """按学习分数选取来源多样的短证据，不使用 gold。"""
    ordered = sorted(
        range(len(hits)),
        key=lambda position: (
            float(predictions[position]),
            hits[position].score,
            -position,
        ),
        reverse=True,
    )
    selected: list[SearchHit] = []
    selected_positions: set[int] = set()
    source_counts: dict[str, int] = defaultdict(int)
    for position in ordered:
        hit = hits[position]
        if source_counts[hit.source] >= max(1, int(max_per_source)):
            continue
        selected.append(hit)
        selected_positions.add(position)
        source_counts[hit.source] += 1
        if len(selected) >= max(1, int(top_k)):
            return selected
    for position in ordered:
        if position in selected_positions:
            continue
        selected.append(hits[position])
        if len(selected) >= max(1, int(top_k)):
            break
    return selected


def _linear_reranker_cross_validation(
    groups: list[dict[str, Any]],
    *,
    top_k: int,
    folds: int = 4,
) -> dict[str, Any]:
    """按整道题分折评估，禁止同题候选同时进入拟合和测试。"""
    groups = [group for group in groups if len(group["hits"])]
    if not groups:
        return {
            "sample_count": 0,
            "unique_question_count": 0,
            "duplicate_question_rows": 0,
            "folds": 0,
            "top_k": int(top_k),
            "heldout_evidence_coverage": 0.0,
            "baseline_top3_evidence_coverage": 0.0,
            "gain_vs_top3": 0.0,
            "regressed_samples": 0,
            "fold_evidence_coverage": [],
            "by_type": {},
            "positive_candidate_rate": 0.0,
            "model": None,
        }
    group_identities = [
        (str(group["type"]), str(group.get("group_key", f"row-{index}")))
        for index, group in enumerate(groups)
    ]
    unique_identities = list(dict.fromkeys(group_identities))
    fold_count = (
        min(max(2, int(folds)), len(unique_identities))
        if len(unique_identities) > 1
        else 1
    )
    fold_assignments: dict[int, int] = {}
    by_type_indices: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        by_type_indices[str(group["type"])].append(index)
    identity_folds = {
        identity: rank % fold_count
        for rank, identity in enumerate(unique_identities)
    }
    for index, identity in enumerate(group_identities):
        fold_assignments[index] = identity_folds[identity]

    heldout_rows: list[dict[str, Any]] = []
    fold_coverages: list[float] = []
    for fold in range(fold_count):
        train_groups = [
            group
            for index, group in enumerate(groups)
            if fold_assignments[index] != fold
        ]
        test_groups = [
            group
            for index, group in enumerate(groups)
            if fold_assignments[index] == fold
        ]
        if not train_groups or not test_groups:
            continue
        model = _fit_linear_reranker(train_groups)
        fold_values: list[float] = []
        for group in test_groups:
            predictions = _predict_linear_reranker(model, group["features"])
            selected = _select_linear_hits(
                group["hits"],
                predictions,
                top_k=top_k,
            )
            coverage = evidence_coverage(group["expected"], selected)
            fold_values.append(coverage)
            heldout_rows.append({
                "type": group["type"],
                "coverage": coverage,
                "baseline": group["baseline"],
            })
        fold_coverages.append(_mean(fold_values))

    full_model = _fit_linear_reranker(groups)
    heldout_coverage = _mean([row["coverage"] for row in heldout_rows])
    baseline_coverage = _mean([row["baseline"] for row in heldout_rows])
    positive_candidates = sum(
        int((np.asarray(group["labels"]) > 0.0).sum()) for group in groups
    )
    candidate_count = sum(len(group["labels"]) for group in groups)
    return {
        "sample_count": len(heldout_rows),
        "unique_question_count": len(unique_identities),
        "duplicate_question_rows": len(groups) - len(unique_identities),
        "folds": fold_count,
        "top_k": int(top_k),
        "heldout_evidence_coverage": heldout_coverage,
        "baseline_top3_evidence_coverage": baseline_coverage,
        "gain_vs_top3": heldout_coverage - baseline_coverage,
        "regressed_samples": sum(
            row["coverage"] < row["baseline"] for row in heldout_rows
        ),
        "fold_evidence_coverage": fold_coverages,
        "by_type": {
            question_type: {
                "count": sum(row["type"] == question_type for row in heldout_rows),
                "heldout_evidence_coverage": _mean([
                    row["coverage"]
                    for row in heldout_rows
                    if row["type"] == question_type
                ]),
                "baseline_top3_evidence_coverage": _mean([
                    row["baseline"]
                    for row in heldout_rows
                    if row["type"] == question_type
                ]),
            }
            for question_type in sorted(by_type_indices)
        },
        "positive_candidate_rate": (
            positive_candidates / candidate_count if candidate_count else 0.0
        ),
        "model": {
            "feature_names": list(ANSWERABILITY_FEATURE_NAMES),
            "mean": [float(value) for value in np.asarray(full_model["mean"])],
            "scale": [float(value) for value in np.asarray(full_model["scale"])],
            "weights": [
                float(value) for value in np.asarray(full_model["weights"])
            ],
            "intercept": float(full_model["intercept"]),
        },
    }


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
    aligned_sibling_expansion: bool = False,
    packing_top_k: int = 8,
    packing_snippet_chars: int = 140,
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
    linear_groups: list[dict[str, Any]] = []
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
                include_aligned_siblings=aligned_sibling_expansion,
            )
        reranked_hits = rerank_answerable_hits(
            question,
            candidate_hits,
            top_k=top_k,
            baseline_hits=baseline_hits[:1],
        )
        snippet_hits = extract_answerable_snippets(
            question,
            candidate_hits,
            max_chars=packing_snippet_chars,
        )
        baseline_snippets = extract_answerable_snippets(
            question,
            baseline_hits[:1],
            max_chars=packing_snippet_chars,
        )
        packed_hits = rerank_answerable_hits(
            question,
            snippet_hits,
            top_k=packing_top_k,
            baseline_hits=baseline_snippets,
        )
        baseline = _retrieval_stats(question, expected, baseline_hits)
        reranked = _retrieval_stats(question, expected, reranked_hits)
        corpus_coverage = _presence_coverage(expected, corpus_presence)
        candidate_coverage = evidence_coverage(expected, raw_candidate_hits)
        expanded_coverage = evidence_coverage(expected, candidate_hits)
        snippet_pool_coverage = evidence_coverage(expected, snippet_hits)
        packed_coverage = evidence_coverage(expected, packed_hits)
        snippet_features = np.asarray(
            answerability_feature_rows(question, snippet_hits),
            dtype=np.float64,
        )
        snippet_labels = np.asarray(
            [evidence_coverage(expected, [hit]) for hit in snippet_hits],
            dtype=np.float64,
        )
        linear_groups.append({
            "row_index": row_index,
            "type": question_type,
            "group_key": compact_text(question),
            "expected": expected,
            "hits": snippet_hits,
            "features": snippet_features,
            "labels": snippet_labels,
            "baseline": reranked["evidence_coverage"],
        })
        baseline_rows.append(baseline)
        reranked_rows.append(reranked)
        per_type[question_type]["baseline"].append(baseline)
        per_type[question_type]["reranked"].append(reranked)

        stage_coverages = {
            "corpus": corpus_coverage,
            "baseline_top3": baseline["evidence_coverage"],
            "candidate_pool": candidate_coverage,
            "expanded_pool": expanded_coverage,
            "snippet_pool": snippet_pool_coverage,
            f"packed_top{packing_top_k}": packed_coverage,
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
    linear_reranker_cv = _linear_reranker_cross_validation(
        linear_groups,
        top_k=packing_top_k,
    )
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
        "snippet_compression_gap": (
            funnel_summary["expanded_pool_evidence_coverage"]
            - funnel_summary["snippet_pool_evidence_coverage"]
        ),
        "packing_selection_gap": (
            funnel_summary["snippet_pool_evidence_coverage"]
            - funnel_summary[f"packed_top{packing_top_k}_evidence_coverage"]
        ),
        "packed_gain_vs_top3": (
            funnel_summary[f"packed_top{packing_top_k}_evidence_coverage"]
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
        "aligned_sibling_expansion": bool(aligned_sibling_expansion),
        "packing_top_k": int(packing_top_k),
        "packing_snippet_chars": int(packing_snippet_chars),
        "linear_reranker_cv": linear_reranker_cv,
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
