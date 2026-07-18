"""QA 检索核心的本地单元测试。"""

from __future__ import annotations

import pytest

from common.environments.qa_search_core import (
    LocalMarkdownIndex,
    QASearchRunner,
    qa_reward_diagnostics,
    qa_type_from_text,
)
from common.rewards.qa_reward import FORMAT_PENALTY, qa_rule_reward_fn


@pytest.fixture
def runner(tmp_path):
    (tmp_path / "process.md").write_text(
        "# 离子注入\n\n离子注入系统由离子源、分析磁场、加速器、扫描系统和反应室组成。",
        encoding="utf-8",
    )
    (tmp_path / "safety.md").write_text(
        "# 登高安全\n\n移动脚手架前必须确认架上无人，施工人员应佩戴安全带。",
        encoding="utf-8",
    )
    index = LocalMarkdownIndex(tmp_path, chunk_chars=200)
    return QASearchRunner(
        index,
        qa_rule_reward_fn,
        top_k=2,
        max_searches=2,
        max_result_chars=300,
    )


def test_chinese_retrieval_returns_relevant_source(runner):
    hits = runner.index.search("离子注入系统组成")
    assert hits
    assert hits[0].source == "process.md"
    assert "离子源" in hits[0].text


def test_search_then_final_answer(runner):
    metadata = {
        "query": "题目：离子注入系统的组成",
        "expected_answer": "[short] 离子源 ||| 分析磁场 ||| 加速器",
        "searches": 0,
    }
    search_result = runner.process_turn(
        [{"role": "assistant", "content": "<search>离子注入 组成</search>"}],
        metadata,
    )
    assert search_result.reward == 0.0
    assert not search_result.terminated
    assert search_result.metadata["searches"] == 1
    assert "process.md" in search_result.observation["content"]

    final_result = runner.process_turn(
        [
            {"role": "assistant", "content": "<search>离子注入 组成</search>"},
            search_result.observation,
            {
                "role": "assistant",
                "content": "<answer>\\boxed{离子源; 分析磁场; 加速器}</answer>",
            },
        ],
        search_result.metadata,
    )
    assert final_result.reward == pytest.approx(1.0)
    assert final_result.terminated
    assert final_result.metadata is None


def test_invalid_format_gets_one_correction(runner):
    correction = runner.process_turn(
        [{"role": "assistant", "content": "我不知道答案"}],
        {"query": "题目：测试", "expected_answer": "[single] A", "searches": 0},
    )
    assert correction.reward == 0.0
    assert not correction.terminated
    assert correction.metadata["must_answer"] is True
    assert correction.metadata["correction_used"] is True

    result = runner.process_turn(
        [{"role": "assistant", "content": "仍然没有按格式作答"}],
        correction.metadata,
    )
    assert result.reward == FORMAT_PENALTY
    assert result.terminated


def test_search_limit_requires_final_answer(runner):
    correction = runner.process_turn(
        [{"role": "assistant", "content": "<search>继续搜索</search>"}],
        {"query": "题目：测试", "expected_answer": "[single] A", "searches": 2},
    )
    assert correction.reward == 0.0
    assert not correction.terminated
    assert correction.metadata["correction_used"] is True

    result = runner.process_turn(
        [{"role": "assistant", "content": "<search>仍然搜索</search>"}],
        correction.metadata,
    )
    assert result.reward == FORMAT_PENALTY
    assert result.terminated


def test_last_search_requires_answer(runner):
    result = runner.process_turn(
        [{"role": "assistant", "content": "<search>继续搜索</search>"}],
        {"query": "题目：测试", "expected_answer": "[single] A", "searches": 1},
    )
    assert not result.terminated
    assert result.metadata["searches"] == 2
    assert result.metadata["must_answer"] is True
    assert "最后一次检索" in result.observation["content"]
    assert "\\boxed{答案}" not in result.observation["content"]


def test_format_correction_can_finish_with_boxed_answer(runner):
    correction = runner.process_turn(
        [{"role": "assistant", "content": "<answer>答案是 A</answer>"}],
        {"query": "题目：测试", "expected_answer": "[single] A", "searches": 0},
    )
    result = runner.process_turn(
        [{"role": "assistant", "content": "<answer>依据；\\boxed{A}</answer>"}],
        correction.metadata,
    )
    assert result.reward == pytest.approx(1.0)
    assert result.terminated


def test_empty_search_falls_back_to_question(runner):
    result = runner.process_turn(
        [{"role": "assistant", "content": "<search> </search>"}],
        {
            "query": "题目：移动脚手架时需要注意什么？",
            "expected_answer": "[short] 架上无人",
            "searches": 0,
        },
    )
    assert not result.terminated
    assert "safety.md" in result.observation["content"]


@pytest.mark.parametrize(
    "text,want",
    [
        ("下面是一道单选题。", "single"),
        ("下面是一道多选题。", "multiple"),
        ("下面是一道判断题。", "bool"),
        ("下面是一道填空题。", "fill"),
        ("下面是一道简答题。", "short"),
        ("未知题型", "unknown"),
    ],
)
def test_qa_type_from_text(text, want):
    assert qa_type_from_text(text) == want


def test_reward_diagnostics_reports_type_and_group_variance():
    metrics = qa_reward_diagnostics(
        rewards=[0.0, 0.0, 0.0, 1.0, 0.5],
        prompt_indices=[1, 1, 2, 2, 3],
        question_types=["fill", "fill", "short", "short", "single"],
    )
    assert metrics["qa_type_fill_mean_reward"] == 0.0
    assert metrics["qa_type_fill_zero_rate"] == 1.0
    assert metrics["qa_type_short_mean_reward"] == 0.5
    assert metrics["qa_multi_sample_group_count"] == 2.0
    assert metrics["qa_zero_variance_group_rate"] == 0.5
    assert metrics["qa_effective_group_rate"] == 0.5
