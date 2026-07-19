"""直接答案 SFT 数据构造测试。"""

from __future__ import annotations

from common.data.qa_direct_sft import build_direct_sft_splits
from common.rewards.qa_reward import qa_rule_reward_fn


def _row(index: int, *, expected: str = "[single] A") -> dict[str, str]:
    return {
        "query": (
            "下面是一道单选题。选出唯一正确的选项。\n"
            f"题目：测试问题 {index}\n\n选项：\nA. 对\nB. 错"
        ),
        "expected_answer": expected,
    }


def test_direct_sft_builds_disjoint_grouped_splits_with_full_reward():
    rows = [_row(index) for index in range(80)]
    rows.append(dict(rows[0]))

    train, validation, stats = build_direct_sft_splits(
        rows,
        lambda query: "PROMPT:" + query,
        validation_denominator=5,
    )

    assert train and validation
    assert stats["trajectory_count"] == 80
    assert stats["duplicate_rows_skipped"] == 1
    assert {row["group_key"] for row in train}.isdisjoint(
        row["group_key"] for row in validation
    )
    for trajectory in [*train, *validation]:
        messages = trajectory["messages"]
        assert messages[0]["content"].startswith("PROMPT:")
        reward = qa_rule_reward_fn(
            [messages[0]["content"]],
            [messages[-1]["content"]],
            ["[single] A"],
        )[0]
        assert reward == 1.0


def test_direct_sft_drops_conflicting_and_type_mismatched_groups():
    conflict_a = _row(1, expected="[single] A")
    conflict_b = _row(1, expected="[single] B")
    mismatched = _row(2, expected="[bool] A")

    train, validation, stats = build_direct_sft_splits(
        [conflict_a, conflict_b, mismatched],
        lambda query: query,
        validation_denominator=5,
    )

    assert not train and not validation
    assert stats["conflicting_question_groups_skipped"] == 1
    assert stats["invalid_target_groups_skipped"] == 1
