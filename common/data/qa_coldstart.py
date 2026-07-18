"""从训练集构造与线上 QA 环境一致的多轮 SFT 正轨迹。"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from statistics import fmean
from typing import Any, Callable

from common.environments.qa_search_core import (
    QASearchRunner,
    _question_text,
    build_query_variants,
    compact_text,
    qa_type_from_text,
)
from common.rewards.qa_reward import qa_rule_reward_fn

_EXPECTED = re.compile(r"\s*\[(\w+)]\s*(.*)", re.DOTALL)
_OPEN_TYPES = {"fill", "short"}
_CLOSED_TYPES = {"single", "multiple", "bool"}


def expected_answer_payload(expected: str) -> tuple[str, str]:
    """把数据集 gold 转成能稳定获得满分的 boxed 内容。"""
    match = _EXPECTED.match(str(expected))
    if not match:
        return "unknown", str(expected).strip()
    question_type, raw = match.group(1), match.group(2).strip()
    if question_type not in _OPEN_TYPES:
        return question_type, raw
    items = []
    for item in raw.split("|||"):
        first_alternative = re.split(r"[/／]", item, maxsplit=1)[0].strip()
        if first_alternative:
            items.append(first_alternative)
    return question_type, "; ".join(items)


def normalized_question_key(query: str) -> str:
    return compact_text(_question_text(query))


def deterministic_search_query(query: str) -> str:
    """只从题干提取一个可在线复现的搜索动作，不使用 gold。"""
    question = _question_text(query)
    variants = build_query_variants(question)
    candidates = [variant for variant in variants[1:] if len(compact_text(variant)) >= 3]
    selected = candidates[0] if candidates else question
    return selected.strip()[:160]


def _build_one_trajectory(
    row: dict[str, Any],
    runner: QASearchRunner,
    render_prompt: Callable[[str], str],
) -> tuple[dict[str, Any] | None, float]:
    query = str(row["query"])
    expected = str(row["expected_answer"])
    question_type, answer_payload = expected_answer_payload(expected)
    if question_type not in _OPEN_TYPES | _CLOSED_TYPES or not answer_payload:
        return None, 0.0

    search_query = deterministic_search_query(query)
    search_action = f"<search>{search_query}</search>"
    search_result = runner.process_turn(
        [{"role": "assistant", "content": search_action}],
        {
            "query": query,
            "expected_answer": expected,
            "searches": 0,
            "must_answer": False,
            "correction_used": False,
            "is_training": True,
            "evidence_coverage": 0.0,
            "evidence_reward_total": 0.0,
        },
    )
    if search_result.terminated or search_result.metadata is None:
        return None, 0.0
    evidence_coverage = float(search_result.metadata.get("evidence_coverage", 0.0))
    if question_type in _OPEN_TYPES and evidence_coverage <= 0.0:
        return None, evidence_coverage

    final_answer = (
        "<answer>根据题目与检索结果，"
        f"\\boxed{{{answer_payload}}}</answer>"
    )
    reward = qa_rule_reward_fn([query], [final_answer], [expected])[0]
    if reward != 1.0:
        return None, evidence_coverage

    group_key = normalized_question_key(query)
    return {
        "messages": [
            {"role": "user", "content": render_prompt(query)},
            {"role": "assistant", "content": search_action},
            search_result.observation,
            {"role": "assistant", "content": final_answer},
        ],
        "question_type": question_type,
        "category": "open" if question_type in _OPEN_TYPES else "closed",
        "group_key": group_key,
        "evidence_coverage": evidence_coverage,
    }, evidence_coverage


def _type_targets(total: int, names: tuple[str, ...]) -> dict[str, int]:
    total = max(0, int(total))
    base, remainder = divmod(total, len(names))
    return {
        name: base + int(position < remainder)
        for position, name in enumerate(names)
    }


def build_coldstart_trajectories(
    rows: list[dict[str, Any]],
    runner: QASearchRunner,
    render_prompt: Callable[[str], str],
    *,
    target_open: int,
    target_closed: int,
    max_open_scan: int,
    max_closed_scan: int,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """按题型配额扫描训练题；不足时如实返回，不重复凑数。"""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    duplicate_rows = 0
    seen_keys: set[tuple[str, str]] = set()
    for row in rows:
        question_type = qa_type_from_text(str(row.get("query", "")))
        key = (question_type, normalized_question_key(str(row.get("query", ""))))
        if not key[1] or key in seen_keys:
            duplicate_rows += 1
            continue
        seen_keys.add(key)
        grouped[question_type].append(row)

    rng = random.Random(seed)
    for candidates in grouped.values():
        rng.shuffle(candidates)

    targets = {
        **_type_targets(target_open, ("fill", "short")),
        **_type_targets(target_closed, ("single", "multiple", "bool")),
    }
    # 扫描上限按题型计算，某一题型样本不足时不会吞掉另一题型的预算。
    scan_limits = {
        "fill": max(0, int(max_open_scan)),
        "short": max(0, int(max_open_scan)),
        "single": max(0, int(max_closed_scan)),
        "multiple": max(0, int(max_closed_scan)),
        "bool": max(0, int(max_closed_scan)),
    }
    selected: list[dict[str, Any]] = []
    scanned = Counter()
    selected_counts = Counter()
    open_coverages: list[float] = []
    for question_type in ("fill", "short", "single", "multiple", "bool"):
        for row in grouped.get(question_type, [])[: scan_limits[question_type]]:
            if selected_counts[question_type] >= targets[question_type]:
                break
            scanned[question_type] += 1
            trajectory, coverage = _build_one_trajectory(row, runner, render_prompt)
            if trajectory is None:
                continue
            selected.append(trajectory)
            selected_counts[question_type] += 1
            if question_type in _OPEN_TYPES:
                open_coverages.append(coverage)

    rng.shuffle(selected)
    stats = {
        "row_count": len(rows),
        "unique_question_count": len(seen_keys),
        "duplicate_rows_skipped": duplicate_rows,
        "targets": targets,
        "scan_limits": scan_limits,
        "scanned_by_type": dict(sorted(scanned.items())),
        "selected_by_type": dict(sorted(selected_counts.items())),
        "trajectory_count": len(selected),
        "mean_open_evidence_coverage": (
            fmean(open_coverages) if open_coverages else 0.0
        ),
    }
    return selected, stats


def split_trajectories(
    trajectories: list[dict[str, Any]],
    *,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按类别和规范化题干分组切分，任何同题轨迹不得跨 split。"""
    fraction = max(0.0, min(0.5, float(validation_fraction)))
    rng = random.Random(seed)
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for trajectory in trajectories:
        grouped[str(trajectory["category"])][str(trajectory["group_key"])].append(
            trajectory
        )

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for category in sorted(grouped):
        keys = sorted(grouped[category])
        rng.shuffle(keys)
        validation_groups = (
            max(1, round(len(keys) * fraction)) if len(keys) >= 2 and fraction > 0 else 0
        )
        validation_keys = set(keys[:validation_groups])
        for key in keys:
            target = validation if key in validation_keys else train
            target.extend(grouped[category][key])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation
