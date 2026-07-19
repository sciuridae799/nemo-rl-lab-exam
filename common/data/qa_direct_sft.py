"""用 QA 训练集构造只监督最终答案的直接 SFT 数据。"""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from typing import Any, Callable

from common.data.qa_coldstart import (
    expected_answer_payload,
    normalized_question_key,
)
from common.environments.qa_search_core import compact_text, qa_type_from_text
from common.rewards.qa_reward import qa_rule_reward_fn

_SUPPORTED_TYPES = {"single", "multiple", "bool", "fill", "short"}


def _stable_validation_group(group_key: str, *, denominator: int) -> bool:
    digest = hashlib.sha1(str(group_key).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % max(2, int(denominator)) == 0


def _answer_key(expected: str) -> str:
    question_type, payload = expected_answer_payload(expected)
    return f"{question_type}:{compact_text(payload)}"


def _direct_trajectory(
    row: dict[str, Any],
    render_prompt: Callable[[str], str],
) -> dict[str, Any] | None:
    query = str(row.get("query", "")).strip()
    expected = str(row.get("expected_answer", "")).strip()
    question_type, payload = expected_answer_payload(expected)
    if (
        not query
        or not payload
        or question_type not in _SUPPORTED_TYPES
        or qa_type_from_text(query) != question_type
    ):
        return None

    completion = f"<answer>\\boxed{{{payload}}}</answer>"
    reward = float(qa_rule_reward_fn([query], [completion], [expected])[0])
    if reward != 1.0:
        return None
    return {
        "messages": [
            {"role": "user", "content": render_prompt(query)},
            {"role": "assistant", "content": completion},
        ],
        "question_type": question_type,
        "group_key": normalized_question_key(query),
    }


def build_direct_sft_splits(
    rows: list[dict[str, Any]],
    render_prompt: Callable[[str], str],
    *,
    validation_denominator: int = 20,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """按规范化题干去重并稳定分组，冲突答案整组丢弃。"""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid_question_rows = 0
    for row in rows:
        key = normalized_question_key(str(row.get("query", "")))
        if not key:
            invalid_question_rows += 1
            continue
        grouped[key].append(row)

    trajectories: list[dict[str, Any]] = []
    duplicate_rows = 0
    conflicting_groups = 0
    invalid_target_groups = 0
    for key in sorted(grouped):
        group_rows = grouped[key]
        duplicate_rows += max(0, len(group_rows) - 1)
        answer_keys = {_answer_key(str(row.get("expected_answer", ""))) for row in group_rows}
        if len(answer_keys) != 1:
            conflicting_groups += 1
            continue
        trajectory = _direct_trajectory(group_rows[0], render_prompt)
        if trajectory is None:
            invalid_target_groups += 1
            continue
        trajectories.append(trajectory)

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    denominator = max(2, int(validation_denominator))
    for trajectory in trajectories:
        target = (
            validation
            if _stable_validation_group(
                str(trajectory["group_key"]), denominator=denominator
            )
            else train
        )
        target.append(trajectory)

    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(validation)
    train_counts = Counter(str(row["question_type"]) for row in train)
    validation_counts = Counter(str(row["question_type"]) for row in validation)
    stats = {
        "row_count": len(rows),
        "unique_question_count": len(grouped),
        "duplicate_rows_skipped": duplicate_rows,
        "invalid_question_rows": invalid_question_rows,
        "conflicting_question_groups_skipped": conflicting_groups,
        "invalid_target_groups_skipped": invalid_target_groups,
        "trajectory_count": len(trajectories),
        "train_count": len(train),
        "validation_count": len(validation),
        "validation_denominator": denominator,
        "train_by_type": dict(sorted(train_counts.items())),
        "validation_by_type": dict(sorted(validation_counts.items())),
    }
    return train, validation, stats
