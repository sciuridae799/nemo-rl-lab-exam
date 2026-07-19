"""在不训练模型的情况下比较 QA 文档检索方案。"""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from difflib import SequenceMatcher
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
_TEACHER_SEARCH = re.compile(r"<search>\s*(.*?)\s*</search>", re.IGNORECASE | re.DOTALL)
_TEACHER_ANSWER = re.compile(r"<answer>[\s\S]*?</answer>", re.IGNORECASE)


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


def _teacher_observation(
    question: str,
    search_query: str,
    hits: list[SearchHit],
    *,
    max_chars: int = 1200,
    must_answer: bool,
) -> str:
    """将检索结果格式化为教师模型可读、但不含 gold 的观测。"""
    safe_query = str(search_query).replace("<", "＜").replace(">", "＞")
    parts = [f'<search_results query="{safe_query}">']
    used = 0
    for rank, hit in enumerate(hits, start=1):
        remaining = int(max_chars) - used
        if remaining <= 0:
            break
        body = hit.text[:remaining]
        parts.append(f"\n[{rank}] 来源: {hit.source}\n{body}")
        used += len(body)
    parts.append("\n</search_results>")
    if must_answer:
        parts.append("\n现在必须直接作答，只输出 answer XML 和 \\boxed{}，不得再次检索。")
    else:
        parts.append("\n如证据不足可再检索一次，否则直接作答。")
    return "".join(parts)


def evaluate_llm_teacher_agent(
    rows: list[dict[str, Any]],
    index: LocalMarkdownIndex,
    teacher: Any,
    *,
    input_key: str = "query",
    output_key: str = "expected_answer",
    max_per_type: int = 4,
    seed: int = 42,
    top_k: int = 3,
    candidate_k: int = 80,
    candidate_max_per_source: int = 4,
    query_expansion: bool = True,
    structural_expansion: bool = False,
    aligned_sibling_expansion: bool = False,
    max_searches: int = 2,
    max_result_chars: int = 1200,
) -> dict[str, Any]:
    """用已有指令模型做小规模端到端教师门控。

    教师只看到题干和 BM25 检索观测；期望答案仅在生成完成后交给官方奖励函数
    计算指标，不参与查询、提示或候选选择。该函数不更新权重。
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        question_type, _ = _gold_items(str(row.get(output_key, "")))
        if question_type != "unknown":
            grouped[question_type].append(row)
    rng = random.Random(seed)
    selected_rows: list[dict[str, Any]] = []
    for question_type in ("single", "multiple", "bool", "fill", "short"):
        candidates = list(grouped.get(question_type, []))
        rng.shuffle(candidates)
        selected_rows.extend(candidates[: max(1, int(max_per_type))])

    system_prompt = (
        "你是技术培训考试问答 Agent。只能依据题干和检索资料作答，不要编造或凭常识补全。"
        "每次只能输出一个完整动作：需要资料时只输出 <search>关键词</search>；"
        "最终作答时只输出 <answer>简短依据；\\boxed{答案}</answer>。"
        "严格按题目要求填写字母、逗号或分号，不要输出思考过程。"
    )
    completions: list[str] = []
    questions: list[str] = []
    expected_answers: list[str] = []
    search_counts: list[float] = []
    errors = 0

    for row in selected_rows:
        query = str(row[input_key])
        question = _question_text(query)
        expected = str(row[output_key])
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "题目：" + question},
        ]
        final_text = ""
        searches = 0
        try:
            while searches <= max(0, int(max_searches)):
                force_answer = searches >= max(0, int(max_searches))
                if force_answer:
                    messages.append({
                        "role": "user",
                        "content": "现在只输出最终 <answer>，必须包含 \\boxed{}，不要输出 <search>。",
                    })
                generated = str(
                    teacher.generate_messages(
                        messages,
                        max_new_tokens=192 if force_answer else 64,
                    )
                ).strip()
                final_match = _TEACHER_ANSWER.search(generated)
                search_match = _TEACHER_SEARCH.search(generated)
                if final_match or ("\\boxed" in generated and not search_match):
                    final_text = generated
                    break
                if not search_match or force_answer:
                    messages.extend([
                        {"role": "assistant", "content": generated},
                        {
                            "role": "user",
                            "content": "格式不完整。现在只输出最终 <answer>，必须包含 \\boxed{}，不要解释。",
                        },
                    ])
                    generated = str(
                        teacher.generate_messages(messages, max_new_tokens=192)
                    ).strip()
                    final_text = generated
                    break

                search_query = search_match.group(1).strip() or question
                retrieval_query = search_query + "\n" + question
                baseline_hits = index.search(retrieval_query, top_k=top_k)
                if query_expansion:
                    queries = build_query_variants(question, search_query)
                else:
                    queries = [retrieval_query]
                candidate_hits = index.search_union(
                    queries,
                    candidate_k=candidate_k,
                    max_per_source=candidate_max_per_source,
                )
                if structural_expansion:
                    candidate_hits = index.expand_structural_candidates(
                        question,
                        candidate_hits,
                        include_aligned_siblings=aligned_sibling_expansion,
                    )
                hits = rerank_answerable_hits(
                    question,
                    candidate_hits,
                    top_k=top_k,
                    baseline_hits=baseline_hits,
                )
                searches += 1
                messages.extend([
                    {"role": "assistant", "content": generated},
                    {
                        "role": "user",
                        "content": _teacher_observation(
                            question,
                            search_query,
                            hits,
                            max_chars=max_result_chars,
                            must_answer=searches >= max(0, int(max_searches)),
                        ),
                    },
                ])
        except Exception:
            errors += 1
            final_text = ""
        completions.append(final_text)
        questions.append(query)
        expected_answers.append(expected)
        search_counts.append(float(searches))

    try:
        from common.rewards.qa_judge_reward import qa_judge_reward_fn

        rewards = qa_judge_reward_fn(questions, completions, expected_answers)
    except Exception:
        from common.rewards.qa_reward import qa_rule_reward_fn

        rewards = qa_rule_reward_fn(questions, completions, expected_answers)

    by_type: dict[str, list[float]] = defaultdict(list)
    for row, reward in zip(selected_rows, rewards, strict=False):
        question_type, _ = _gold_items(str(row.get(output_key, "")))
        by_type[question_type].append(float(reward))
    open_rewards = [
        float(reward)
        for row, reward in zip(selected_rows, rewards, strict=False)
        if _gold_items(str(row.get(output_key, "")))[0] in _OPEN_TYPES
    ]
    return {
        "sample_count": len(selected_rows),
        "max_per_type": int(max_per_type),
        "seed": int(seed),
        "mean_reward": _mean([float(value) for value in rewards]),
        "perfect_rate": _mean([float(float(value) >= 1.0) for value in rewards]),
        "format_penalty_count": sum(float(value) < 0.0 for value in rewards),
        "open_positive_count": sum(value > 0.0 for value in open_rewards),
        "open_count": len(open_rewards),
        "mean_searches": _mean(search_counts),
        "errors": errors,
        "by_type": {
            question_type: {
                "count": len(values),
                "mean_reward": _mean(values),
                "perfect_count": sum(value >= 1.0 for value in values),
            }
            for question_type, values in sorted(by_type.items())
        },
    }


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


def _boxed_from_expected(expected: str) -> str:
    match = _EXPECTED.match(str(expected))
    payload = match.group(2).strip() if match else str(expected).strip()
    # 官方规则对简答要点使用 ||| 分隔，模型答案使用分号。
    return "\\boxed{" + payload.replace("|||", ";") + "}"


def evaluate_qa_memory_knn(
    rows: list[dict[str, Any]],
    *,
    input_key: str = "query",
    output_key: str = "expected_answer",
    neighbors: int = 5,
) -> dict[str, Any]:
    """评估训练集内非参数问题记忆的 held-out 上限。

    按规范化题干分组后做稳定 80/20 切分，确保同题重复行不跨折。该诊断只使用
    训练数据，用来判断“相似训练题提示”是否值得进入无训练 A/B。
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:  # pragma: no cover - 远端能力依赖
        raise RuntimeError("QA memory diagnostic 需要 scikit-learn") from exc
    from common.rewards.qa_reward import qa_rule_reward_fn

    typed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        question_type, _ = _gold_items(str(row.get(output_key, "")))
        if question_type in {"single", "multiple", "bool", "fill", "short"}:
            typed[question_type].append(row)

    by_type: dict[str, Any] = {}
    all_top1: list[float] = []
    all_vote: list[float] = []
    all_oracle: list[float] = []
    confidence_rows: list[tuple[float, float]] = []
    for question_type, type_rows in sorted(typed.items()):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in type_rows:
            question = _question_text(str(row[input_key]))
            grouped[compact_text(question)].append(row)
        train_rows: list[dict[str, Any]] = []
        holdout_rows: list[dict[str, Any]] = []
        for group_rows in grouped.values():
            if _stable_holdout(str(group_rows[0][input_key])):
                holdout_rows.extend(group_rows)
            else:
                train_rows.extend(group_rows)
        if not train_rows or not holdout_rows:
            continue

        train_questions = [_question_text(str(row[input_key])) for row in train_rows]
        holdout_questions = [_question_text(str(row[input_key])) for row in holdout_rows]
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=1,
            max_features=80_000,
            sublinear_tf=True,
        )
        train_matrix = vectorizer.fit_transform(train_questions)
        holdout_matrix = vectorizer.transform(holdout_questions)
        similarity_matrix = (holdout_matrix @ train_matrix.T).toarray()

        top1_values: list[float] = []
        vote_values: list[float] = []
        oracle_values: list[float] = []
        similarities: list[float] = []
        for row_index, row in enumerate(holdout_rows):
            similarities_row = similarity_matrix[row_index]
            top_positions = sorted(
                range(len(similarities_row)),
                key=lambda position: (float(similarities_row[position]), -position),
                reverse=True,
            )[: max(1, int(neighbors))]
            predicted_rows = [train_rows[position] for position in top_positions]
            query = str(row[input_key])
            expected = str(row[output_key])
            candidate_rewards = [
                float(
                    qa_rule_reward_fn(
                        [query],
                        [_boxed_from_expected(str(candidate[output_key]))],
                        [expected],
                    )[0]
                )
                for candidate in predicted_rows
            ]
            top1 = candidate_rewards[0]
            oracle = max(candidate_rewards)

            vote_scores: dict[str, float] = defaultdict(float)
            vote_expected: dict[str, str] = {}
            for position in top_positions:
                candidate_expected = str(train_rows[position][output_key])
                answer_key = compact_text(candidate_expected)
                vote_scores[answer_key] += max(0.0, float(similarities_row[position]))
                vote_expected[answer_key] = candidate_expected
            best_vote_key = max(
                vote_scores,
                key=lambda key: (vote_scores[key], key),
            )
            vote = float(
                qa_rule_reward_fn(
                    [query],
                    [_boxed_from_expected(vote_expected[best_vote_key])],
                    [expected],
                )[0]
            )
            similarity = float(similarities_row[top_positions[0]])
            top1_values.append(top1)
            vote_values.append(vote)
            oracle_values.append(oracle)
            similarities.append(similarity)
            confidence_rows.append((similarity, top1))

        by_type[question_type] = {
            "train_count": len(train_rows),
            "holdout_count": len(holdout_rows),
            "unique_question_count": len(grouped),
            "top1_reward": _mean(top1_values),
            "vote_reward": _mean(vote_values),
            "top5_oracle_reward": _mean(oracle_values),
            "mean_top1_similarity": _mean(similarities),
        }
        all_top1.extend(top1_values)
        all_vote.extend(vote_values)
        all_oracle.extend(oracle_values)

    confidence = {}
    for threshold in (0.50, 0.60, 0.70, 0.80, 0.90):
        selected = [reward for similarity, reward in confidence_rows if similarity >= threshold]
        confidence[f"at_{threshold:.2f}"] = {
            "count": len(selected),
            "coverage": len(selected) / len(confidence_rows) if confidence_rows else 0.0,
            "top1_reward": _mean(selected),
        }
    return {
        "sample_count": len(confidence_rows),
        "top1_reward": _mean(all_top1),
        "vote_reward": _mean(all_vote),
        "top5_oracle_reward": _mean(all_oracle),
        "by_type": by_type,
        "confidence": confidence,
    }


def _parse_closed_options(query: str) -> dict[str, str]:
    """解析常见 A./B./C. 选项，不依赖具体题目内容。"""
    match = re.search(r"选项[:：]\s*(.*)", str(query), flags=re.DOTALL)
    if not match:
        return {}
    text = re.sub(r"\s+", " ", match.group(1)).strip()
    pieces = re.split(r"\s+(?=[A-Z][.)、:：])", text)
    options: dict[str, str] = {}
    for piece in pieces:
        item = re.match(r"\s*([A-Z])\s*[.)、:：]\s*(.*)", piece)
        if item:
            options[item.group(1)] = item.group(2).strip()
    return options


def _option_similarity(left: str, right: str) -> float:
    left_text = compact_text(left)
    right_text = compact_text(right)
    if not left_text or not right_text:
        return 0.0
    if left_text == right_text:
        return 1.0
    left_bigrams = {left_text[i : i + 2] for i in range(max(0, len(left_text) - 1))}
    right_bigrams = {right_text[i : i + 2] for i in range(max(0, len(right_text) - 1))}
    jaccard = len(left_bigrams & right_bigrams) / max(1, len(left_bigrams | right_bigrams))
    return max(jaccard, SequenceMatcher(None, left_text, right_text).ratio())


def _closed_answer_letters(expected: str) -> set[str]:
    match = _EXPECTED.match(str(expected))
    payload = match.group(2) if match else str(expected)
    return set(re.findall(r"[A-Z]", payload.upper()))


def evaluate_qa_memory_option_mapping(
    rows: list[dict[str, Any]],
    *,
    input_key: str = "query",
    output_key: str = "expected_answer",
    neighbors: int = 5,
    option_thresholds: tuple[float, ...] = (0.25, 0.40, 0.55),
    vote_thresholds: tuple[float, ...] = (0.30, 0.50, 0.70),
) -> dict[str, Any]:
    """评估相似训练题答案经过选项语义映射后的 held-out 上限。

    只在训练集内部按题干分组切分；验证答案不会进入拟合或候选生成。
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("QA memory mapping 需要 scikit-learn") from exc
    from common.rewards.qa_reward import qa_rule_reward_fn

    closed_types = {"single", "multiple", "bool"}
    typed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        question_type, _ = _gold_items(str(row.get(output_key, "")))
        if question_type in closed_types:
            typed[question_type].append(row)

    overall: dict[str, dict[str, list[float]]] = {
        f"option_{option_threshold:.2f}_vote_{vote_threshold:.2f}": {
            "raw": [],
            "mapped": [],
            "oracle": [],
        }
        for option_threshold in option_thresholds
        for vote_threshold in vote_thresholds
    }

    for question_type, type_rows in sorted(typed.items()):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in type_rows:
            groups[compact_text(_question_text(str(row[input_key])))].append(row)
        train_rows: list[dict[str, Any]] = []
        holdout_rows: list[dict[str, Any]] = []
        for group_rows in groups.values():
            target = holdout_rows if _stable_holdout(str(group_rows[0][input_key])) else train_rows
            target.extend(group_rows)
        if not train_rows or not holdout_rows:
            continue
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=1,
            max_features=80_000,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(
            [_question_text(str(row[input_key])) for row in train_rows]
        )
        for row in holdout_rows:
            query = str(row[input_key])
            expected = str(row[output_key])
            current_options = _parse_closed_options(query)
            similarities = (
                matrix @ vectorizer.transform([_question_text(query)]).T
            ).toarray().ravel()
            positions = sorted(
                range(len(similarities)),
                key=lambda position: (float(similarities[position]), -position),
                reverse=True,
            )[: max(1, int(neighbors))]
            raw_prediction = _boxed_from_expected(str(train_rows[positions[0]][output_key]))
            raw_reward = float(qa_rule_reward_fn([query], [raw_prediction], [expected])[0])
            for option_threshold in option_thresholds:
                weighted: dict[str, float] = defaultdict(float)
                oracle_candidates: list[str] = []
                for position in positions:
                    neighbor = train_rows[position]
                    neighbor_options = _parse_closed_options(str(neighbor[input_key]))
                    neighbor_letters = _closed_answer_letters(
                        str(neighbor[output_key])
                    )
                    mapped_scores: list[tuple[str, float]] = []
                    for letter in neighbor_letters:
                        source = neighbor_options.get(letter)
                        if not source or not current_options:
                            continue
                        target, score = max(
                            (
                                (target_letter, _option_similarity(source, target_text))
                                for target_letter, target_text in current_options.items()
                            ),
                            key=lambda item: item[1],
                            default=("", 0.0),
                        )
                        if score >= option_threshold:
                            mapped_scores.append((target, score))
                    prediction = {letter for letter, _ in mapped_scores}
                    oracle_candidates.append(",".join(sorted(prediction)))
                    for letter, score in mapped_scores:
                        weighted[letter] += max(0.0, float(similarities[position])) * score
                for vote_threshold in vote_thresholds:
                    max_weight = max(weighted.values(), default=0.0)
                    if question_type in {"single", "bool"} and weighted:
                        prediction = {
                            max(weighted, key=lambda letter: (weighted[letter], letter))
                        }
                    else:
                        prediction = {
                            letter
                            for letter, weight in weighted.items()
                            if max_weight > 0.0
                            and weight >= max_weight * vote_threshold
                        }
                    mapped_text = "\\boxed{" + ",".join(sorted(prediction)) + "}"
                    mapped_reward = float(
                        qa_rule_reward_fn([query], [mapped_text], [expected])[0]
                    )
                    oracle_reward = max(
                        float(
                            qa_rule_reward_fn(
                                [query], [f"\\boxed{{{candidate}}}"], [expected]
                            )[0]
                        )
                        for candidate in oracle_candidates
                    ) if oracle_candidates else 0.0
                    key = f"option_{option_threshold:.2f}_vote_{vote_threshold:.2f}"
                    overall[key]["raw"].append(raw_reward)
                    overall[key]["mapped"].append(mapped_reward)
                    overall[key]["oracle"].append(oracle_reward)

    summary = {
        key: {
            "sample_count": len(values["mapped"]),
            "raw_reward": _mean(values["raw"]),
            "mapped_reward": _mean(values["mapped"]),
            "oracle_reward": _mean(values["oracle"]),
            "gain": _mean(values["mapped"]) - _mean(values["raw"]),
        }
        for key, values in sorted(overall.items())
    }
    best = max(summary.items(), key=lambda item: item[1]["mapped_reward"], default=("", {}))
    return {"settings": summary, "best": {"name": best[0], **best[1]}}


def evaluate_llm_candidate_reranker(
    rows: list[dict[str, Any]],
    index: LocalMarkdownIndex,
    reranker: Any,
    *,
    input_key: str = "query",
    output_key: str = "expected_answer",
    max_per_type: int = 16,
    seed: int = 42,
    top_k: int = 3,
    candidate_k: int = 80,
    candidate_max_per_source: int = 4,
    shortlist_size: int = 18,
) -> dict[str, Any]:
    """让指令模型只看问题和候选，评估语义 Top-3 重排的训练集门控。"""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        question_type, _ = _gold_items(str(row.get(output_key, "")))
        if question_type in _OPEN_TYPES:
            question = _question_text(str(row.get(input_key, "")))
            grouped[question_type][compact_text(question)].append(row)
    rng = random.Random(seed)
    selected_rows: list[dict[str, Any]] = []
    for question_type in sorted(_OPEN_TYPES):
        # 只从规范化题干的稳定 held-out 折抽样，避免同题重复行泄漏到门控结果。
        candidates = [
            row
            for question, group_rows in grouped.get(question_type, {}).items()
            if _stable_holdout(question)
            for row in group_rows
        ]
        rng.shuffle(candidates)
        selected_rows.extend(candidates[: max(1, int(max_per_type))])

    default_values: list[float] = []
    candidate_values: list[float] = []
    llm_values: list[float] = []
    by_type: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"default": [], "candidate": [], "llm": []}
    )
    empty_selections = 0
    mean_selected: list[float] = []
    raw_examples: list[dict[str, Any]] = []

    for row in selected_rows:
        question = _question_text(str(row[input_key]))
        expected = str(row[output_key])
        question_type = _gold_items(expected)[0]
        baseline_hits = index.search(question, top_k=top_k)
        candidates = index.search_union(
            build_query_variants(question),
            candidate_k=candidate_k,
            max_per_source=candidate_max_per_source,
        )
        candidates = index.expand_structural_candidates(
            question,
            candidates,
            include_aligned_siblings=True,
        )
        default_hits = rerank_answerable_hits(
            question,
            candidates,
            top_k=max(8, top_k),
            baseline_hits=baseline_hits[:1],
            weights=_ANSWERABILITY_WEIGHT_GRID["default"],
        )
        role_hits = rerank_answerable_hits(
            question,
            candidates,
            top_k=max(8, top_k),
            baseline_hits=baseline_hits[:1],
            weights=_ANSWERABILITY_WEIGHT_GRID["answer_role_strict"],
        )
        shortlist: list[SearchHit] = []
        seen: set[tuple[str, str]] = set()
        for hit in [*default_hits, *candidates[:8], *role_hits]:
            key = (hit.source, hit.text)
            if key in seen:
                continue
            seen.add(key)
            shortlist.append(hit)
            if len(shortlist) >= max(top_k, int(shortlist_size)):
                break

        prompt_candidates: list[tuple[str, str]] = []
        for hit in shortlist:
            snippets = extract_answerable_snippets(
                question,
                [hit],
                max_chars=220,
            )
            prompt_candidates.append(
                (hit.source, snippets[0].text if snippets else hit.text[:220])
            )
        selected_positions, raw = reranker.select(
            question,
            prompt_candidates,
            limit=top_k,
        )
        empty_selections += int(not selected_positions)
        chosen = [shortlist[position] for position in selected_positions]
        chosen_keys = {(hit.source, hit.text) for hit in chosen}
        for hit in default_hits:
            key = (hit.source, hit.text)
            if key not in chosen_keys:
                chosen.append(hit)
                chosen_keys.add(key)
            if len(chosen) >= top_k:
                break
        chosen = chosen[:top_k]

        default_coverage = evidence_coverage(expected, default_hits[:top_k])
        candidate_coverage = evidence_coverage(expected, shortlist)
        llm_coverage = evidence_coverage(expected, chosen)
        default_values.append(default_coverage)
        candidate_values.append(candidate_coverage)
        llm_values.append(llm_coverage)
        mean_selected.append(float(len(selected_positions)))
        by_type[question_type]["default"].append(default_coverage)
        by_type[question_type]["candidate"].append(candidate_coverage)
        by_type[question_type]["llm"].append(llm_coverage)
        if len(raw_examples) < 8:
            raw_examples.append(
                {
                    "type": question_type,
                    "raw": raw[:120],
                    "selected": selected_positions,
                    "default_coverage": default_coverage,
                    "candidate_coverage": candidate_coverage,
                    "llm_coverage": llm_coverage,
                }
            )

    return {
        "sample_count": len(selected_rows),
        "shortlist_size": int(shortlist_size),
        "default_top3_evidence_coverage": _mean(default_values),
        "shortlist_evidence_coverage": _mean(candidate_values),
        "llm_top3_evidence_coverage": _mean(llm_values),
        "llm_gain_vs_default": _mean(llm_values) - _mean(default_values),
        "selection_gap": _mean(candidate_values) - _mean(llm_values),
        "empty_selection_count": empty_selections,
        "mean_selected_count": _mean(mean_selected),
        "by_type": {
            question_type: {
                "count": len(values["default"]),
                "default_top3_evidence_coverage": _mean(values["default"]),
                "shortlist_evidence_coverage": _mean(values["candidate"]),
                "llm_top3_evidence_coverage": _mean(values["llm"]),
            }
            for question_type, values in sorted(by_type.items())
        },
        "raw_examples": raw_examples,
    }


def evaluate_llm_query_rewrite(
    rows: list[dict[str, Any]],
    index: LocalMarkdownIndex,
    reranker: Any,
    *,
    input_key: str = "query",
    output_key: str = "expected_answer",
    max_per_type: int = 8,
    seed: int = 42,
    top_k: int = 3,
    candidate_k: int = 80,
    candidate_max_per_source: int = 4,
    shortlist_size: int = 18,
) -> dict[str, Any]:
    """评估问题改写能否先提高召回，再由指令模型保留到 Top-K。

    指令模型只接收题干和候选文本；期望答案仅用于训练集离线覆盖率测量。
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        question_type, _ = _gold_items(str(row.get(output_key, "")))
        if question_type in _OPEN_TYPES:
            grouped[question_type].append(row)
    rng = random.Random(seed)
    selected_rows: list[dict[str, Any]] = []
    for question_type in sorted(_OPEN_TYPES):
        candidates = list(grouped.get(question_type, []))
        rng.shuffle(candidates)
        selected_rows.extend(candidates[: max(1, int(max_per_type))])

    values: dict[str, list[float]] = defaultdict(list)
    by_type: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    rewrite_counts: list[float] = []
    empty_rewrites = 0
    empty_selections = 0
    raw_examples: list[dict[str, Any]] = []

    for row in selected_rows:
        question = _question_text(str(row[input_key]))
        expected = str(row[output_key])
        question_type = _gold_items(expected)[0]
        baseline_queries = build_query_variants(question)
        baseline_candidates = index.search_union(
            baseline_queries,
            candidate_k=candidate_k,
            max_per_source=candidate_max_per_source,
        )
        baseline_hits = rerank_answerable_hits(
            question,
            baseline_candidates,
            top_k=top_k,
            baseline_hits=index.search(question, top_k=1),
        )

        rewritten_queries, raw_rewrite = reranker.rewrite_queries(
            question,
            limit=3,
        )
        empty_rewrites += int(not rewritten_queries)
        rewrite_counts.append(float(len(rewritten_queries)))
        combined_queries = list(baseline_queries)
        seen_queries = {compact_text(query) for query in combined_queries}
        for query in rewritten_queries:
            key = compact_text(query)
            if key and key not in seen_queries:
                seen_queries.add(key)
                combined_queries.append(query)

        rewritten_candidates = index.search_union(
            combined_queries,
            candidate_k=candidate_k,
            max_per_source=candidate_max_per_source,
        )
        expanded_candidates = index.expand_structural_candidates(
            question,
            rewritten_candidates,
            include_aligned_siblings=True,
        )
        default_hits = rerank_answerable_hits(
            question,
            expanded_candidates,
            top_k=max(top_k, shortlist_size),
            baseline_hits=index.search(question, top_k=1),
        )
        shortlist = default_hits[: max(top_k, int(shortlist_size))]
        snippet_pool = extract_answerable_snippets(
            question,
            expanded_candidates,
            max_chars=140,
        )
        packed_top6 = rerank_answerable_hits(
            question,
            snippet_pool,
            top_k=6,
            baseline_hits=snippet_pool[:1],
        )
        packed_top8 = rerank_answerable_hits(
            question,
            snippet_pool,
            top_k=8,
            baseline_hits=snippet_pool[:1],
        )
        prompt_candidates: list[tuple[str, str]] = []
        for hit in shortlist:
            snippets = extract_answerable_snippets(
                question,
                [hit],
                max_chars=220,
            )
            prompt_candidates.append(
                (hit.source, snippets[0].text if snippets else hit.text[:220])
            )
        selected_positions, raw_rerank = reranker.select(
            question,
            prompt_candidates,
            limit=top_k,
        )
        empty_selections += int(not selected_positions)
        chosen = [shortlist[position] for position in selected_positions]
        chosen_keys = {(hit.source, hit.text) for hit in chosen}
        for hit in default_hits:
            key = (hit.source, hit.text)
            if key not in chosen_keys:
                chosen.append(hit)
                chosen_keys.add(key)
            if len(chosen) >= top_k:
                break
        chosen = chosen[:top_k]

        coverages = {
            "baseline_top3": evidence_coverage(expected, baseline_hits),
            "baseline_pool": evidence_coverage(expected, baseline_candidates),
            "rewrite_pool": evidence_coverage(expected, rewritten_candidates),
            "expanded_pool": evidence_coverage(expected, expanded_candidates),
            "shortlist": evidence_coverage(expected, shortlist),
            "packed_top6": evidence_coverage(expected, packed_top6),
            "packed_top8": evidence_coverage(expected, packed_top8),
            "llm_top3": evidence_coverage(expected, chosen),
        }
        for name, coverage in coverages.items():
            values[name].append(coverage)
            by_type[question_type][name].append(coverage)
        if len(raw_examples) < 8:
            raw_examples.append(
                {
                    "type": question_type,
                    "rewritten_queries": rewritten_queries,
                    "raw_rewrite": raw_rewrite[:240],
                    "raw_rerank": raw_rerank[:120],
                    **{f"{name}_coverage": value for name, value in coverages.items()},
                }
            )

    summary = {f"{name}_evidence_coverage": _mean(stage) for name, stage in values.items()}
    return {
        "sample_count": len(selected_rows),
        "max_per_type": int(max_per_type),
        "seed": int(seed),
        "top_k": int(top_k),
        "shortlist_size": int(shortlist_size),
        **summary,
        "rewrite_pool_gain": summary.get("rewrite_pool_evidence_coverage", 0.0)
        - summary.get("baseline_pool_evidence_coverage", 0.0),
        "llm_gain_vs_baseline_top3": summary.get("llm_top3_evidence_coverage", 0.0)
        - summary.get("baseline_top3_evidence_coverage", 0.0),
        "final_selection_gap": summary.get("expanded_pool_evidence_coverage", 0.0)
        - summary.get("llm_top3_evidence_coverage", 0.0),
        "empty_rewrite_count": empty_rewrites,
        "mean_rewrite_count": _mean(rewrite_counts),
        "empty_selection_count": empty_selections,
        "by_type": {
            question_type: {
                "count": len(type_values.get("baseline_top3", [])),
                **{
                    f"{name}_evidence_coverage": _mean(stage)
                    for name, stage in sorted(type_values.items())
                },
            }
            for question_type, type_values in sorted(by_type.items())
        },
        "raw_examples": raw_examples,
    }


def evaluate_llm_binary_reranker(
    rows: list[dict[str, Any]],
    index: LocalMarkdownIndex,
    reranker: Any,
    *,
    input_key: str = "query",
    output_key: str = "expected_answer",
    max_per_type: int = 4,
    seed: int = 42,
    top_k: int = 3,
    candidate_k: int = 80,
    candidate_max_per_source: int = 4,
    shortlist_size: int = 10,
) -> dict[str, Any]:
    """用逐候选二分类 cross-encoder 做训练集 held-out 检索门控。"""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        question_type, _ = _gold_items(str(row.get(output_key, "")))
        if question_type in _OPEN_TYPES:
            question = _question_text(str(row.get(input_key, "")))
            grouped[question_type][compact_text(question)].append(row)
    rng = random.Random(seed)
    selected_rows: list[dict[str, Any]] = []
    for question_type in sorted(_OPEN_TYPES):
        # 按规范化题干固定 held-out 折，避免同题重复行泄漏到门控结果。
        candidates = [
            row
            for question, group_rows in grouped.get(question_type, {}).items()
            if _stable_holdout(question)
            for row in group_rows
        ]
        rng.shuffle(candidates)
        selected_rows.extend(candidates[: max(1, int(max_per_type))])

    values: dict[str, list[float]] = defaultdict(list)
    by_type: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    positive_predictions = 0
    score_count = 0
    raw_examples: list[dict[str, Any]] = []

    for row in selected_rows:
        question = _question_text(str(row[input_key]))
        expected = str(row[output_key])
        baseline_hits = index.search(question, top_k=top_k)
        candidates = index.search_union(
            build_query_variants(question),
            candidate_k=candidate_k,
            max_per_source=candidate_max_per_source,
        )
        candidates = index.expand_structural_candidates(
            question,
            candidates,
            include_aligned_siblings=True,
        )
        deterministic = rerank_answerable_hits(
            question,
            candidates,
            top_k=max(top_k, int(shortlist_size)),
            baseline_hits=baseline_hits[:1],
        )[: max(top_k, int(shortlist_size))]
        scored: list[tuple[float, int, SearchHit, str]] = []
        for position, hit in enumerate(deterministic):
            score, raw = reranker.judge_candidate(question, hit.source, hit.text[:260])
            score_count += 1
            positive_predictions += int(score > 0.5)
            scored.append((score, position, hit, raw))
        ordered = sorted(
            scored,
            key=lambda item: (item[0], item[2].score, -item[1]),
            reverse=True,
        )
        chosen: list[SearchHit] = []
        source_counts: dict[str, int] = defaultdict(int)
        for _score, _, hit, _ in ordered:
            if source_counts[hit.source] >= 2:
                continue
            chosen.append(hit)
            source_counts[hit.source] += 1
            if len(chosen) >= top_k:
                break
        if len(chosen) < top_k:
            chosen.extend(hit for _, _, hit, _ in ordered if hit not in chosen)
            chosen = chosen[:top_k]
        coverages = {
            "baseline_top3": evidence_coverage(expected, baseline_hits),
            "candidate_pool": evidence_coverage(expected, candidates),
            "binary_top3": evidence_coverage(expected, chosen),
        }
        for name, coverage in coverages.items():
            values[name].append(coverage)
            question_type = _gold_items(expected)[0]
            by_type[question_type][name].append(coverage)
        if len(raw_examples) < 6:
            raw_examples.append(
                {
                    "type": _gold_items(expected)[0],
                    "baseline_top3": coverages["baseline_top3"],
                    "candidate_pool": coverages["candidate_pool"],
                    "binary_top3": coverages["binary_top3"],
                    "positive_count": sum(score > 0.5 for score, *_ in scored),
                }
            )

    summary = {f"{name}_evidence_coverage": _mean(stage) for name, stage in values.items()}
    return {
        "sample_count": len(selected_rows),
        "max_per_type": int(max_per_type),
        "shortlist_size": int(shortlist_size),
        **summary,
        "gain_vs_baseline": summary.get("binary_top3_evidence_coverage", 0.0)
        - summary.get("baseline_top3_evidence_coverage", 0.0),
        "positive_prediction_rate": positive_predictions / score_count if score_count else 0.0,
        "score_count": score_count,
        "by_type": {
            question_type: {
                "count": len(type_values.get("binary_top3", [])),
                **{
                    f"{name}_evidence_coverage": _mean(stage)
                    for name, stage in sorted(type_values.items())
                },
            }
            for question_type, type_values in sorted(by_type.items())
        },
        "raw_examples": raw_examples,
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
