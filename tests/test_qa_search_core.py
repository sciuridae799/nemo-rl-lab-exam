"""QA 检索核心的本地单元测试。"""

from __future__ import annotations

import pytest

from common.environments.qa_search_core import (
    LocalMarkdownIndex,
    QASearchRunner,
    SearchHit,
    build_query_variants,
    evidence_progress_coverage,
    extract_answerable_snippets,
    qa_loss_multiplier,
    qa_reward_diagnostics,
    qa_type_from_text,
    question_copy_score,
    rerank_answerable_hits,
    source_family_key,
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


def test_train_only_evidence_progress_reward_does_not_change_observation(runner):
    shaped_runner = QASearchRunner(
        runner.index,
        qa_rule_reward_fn,
        top_k=2,
        max_searches=2,
        max_result_chars=300,
        evidence_reward_scale=0.1,
    )
    action = [{"role": "assistant", "content": "<search>离子注入 组成</search>"}]
    base_metadata = {
        "query": "下面是一道简答题。\n题目：离子注入系统的组成",
        "expected_answer": "[short] 离子源 ||| 分析磁场 ||| 加速器",
        "searches": 0,
    }

    train_result = shaped_runner.process_turn(
        action,
        {**base_metadata, "is_training": True},
    )
    validation_result = shaped_runner.process_turn(
        action,
        {**base_metadata, "is_training": False},
    )

    assert train_result.reward == 0.0
    assert validation_result.reward == 0.0
    assert train_result.observation == validation_result.observation
    assert train_result.metadata["evidence_coverage"] == pytest.approx(1.0)

    repeated = shaped_runner.process_turn(action, train_result.metadata)
    assert repeated.reward == 0.0

    shaped_final = shaped_runner.process_turn(
        [{"role": "assistant", "content": "<answer>\\boxed{无关答案}</answer>"}],
        train_result.metadata,
    )
    validation_final = shaped_runner.process_turn(
        [{"role": "assistant", "content": "<answer>\\boxed{无关答案}</answer>"}],
        validation_result.metadata,
    )
    assert shaped_final.reward == pytest.approx(0.1)
    assert validation_final.reward == 0.0

    invalid_final = shaped_runner.process_turn(
        [{"role": "assistant", "content": "仍未按格式作答"}],
        {**train_result.metadata, "correction_used": True},
    )
    assert invalid_final.reward == FORMAT_PENALTY


def test_evidence_progress_does_not_reward_words_already_in_question():
    question = "请画出一级疏散集合点的路线图"
    hits = [
        SearchHit(
            "空白试卷.md",
            "简答题：请画出一级疏散集合点的路线图",
            1.0,
        )
    ]

    coverage = evidence_progress_coverage(
        question,
        "[short] 一级疏散集合点 ||| 北门",
        hits,
    )

    assert coverage == 0.0


@pytest.mark.parametrize(
    ("question", "is_training", "want"),
    [
        ("下面是一道填空题。", True, 0.25),
        ("下面是一道简答题。", True, 0.25),
        ("下面是一道单选题。", True, 1.0),
        ("下面是一道简答题。", False, 1.0),
    ],
)
def test_open_loss_multiplier_only_applies_to_training(
    question, is_training, want
):
    assert (
        qa_loss_multiplier(
            question,
            is_training=is_training,
            open_loss_multiplier=0.25,
        )
        == want
    )


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


def test_answerability_rerank_prefers_filled_evidence_over_blank_exam():
    question = "SERVER ROOM 通过【1】与Clean room进行连接"
    blank_exam = SearchHit(
        "试卷.md",
        "1. 填空题（每题4分）\n3. SERVER ROOM 通过 与Clean room进行连接",
        100.0,
    )
    answer_manual = SearchHit(
        "操作手册.md",
        "SERVER ROOM 通过 SQL server 与 Clean room 进行连接。",
        80.0,
    )

    ranked = rerank_answerable_hits(
        question,
        [blank_exam, answer_manual],
        top_k=2,
    )

    assert question_copy_score(question, blank_exam.text) == 1.0
    assert question_copy_score(question, answer_manual.text) == 0.0
    assert ranked[0].source == "操作手册.md"


def test_query_variants_split_ascii_and_cloze_context():
    variants = build_query_variants("GC-MS 在【1】进行检测", "GC-MS 检测")
    assert any("GC" in variant and "MS" in variant for variant in variants)
    assert any("进行检测" in variant for variant in variants)


def test_ascii_prefix_expansion_recovers_morphological_match(tmp_path):
    (tmp_path / "litho.md").write_text(
        "# Lithography\n\nLithography processing follows the standard process flow.",
        encoding="utf-8",
    )
    index = LocalMarkdownIndex(tmp_path, expand_ascii_tokens=True)
    hits = index.search("litho process", top_k=1)
    assert hits and hits[0].source == "litho.md"


def test_source_family_key_links_exam_and_answer_files():
    assert source_family_key("部门/设备培训-试卷.md") == source_family_key(
        "部门/设备培训-答案.md"
    )


def test_structural_expansion_adds_adjacent_and_sibling_evidence(tmp_path):
    question = "设备通过【1】连接洁净室"
    (tmp_path / "模块-试卷.md").write_text(
        "# 试卷\n\n"
        + f"填空题：{question}\n"
        + "题面说明" * 15
        + "\n\n# 下一页\n\n相邻页说明该设备通过数据总线连接洁净室。",
        encoding="utf-8",
    )
    (tmp_path / "模块-答案.md").write_text(
        "# 参考答案\n\n设备通过数据总线连接洁净室。",
        encoding="utf-8",
    )
    index = LocalMarkdownIndex(tmp_path, chunk_chars=160)
    exam_chunk = next(
        chunk for chunk in index.chunks if chunk.source.endswith("试卷.md") and "填空题" in chunk.text
    )

    expanded = index.expand_structural_candidates(
        question,
        [SearchHit(exam_chunk.source, exam_chunk.text, 10.0)],
    )

    assert any("相邻页说明" in hit.text for hit in expanded)
    assert any(hit.source.endswith("答案.md") for hit in expanded)


def test_answerable_snippet_keeps_relevant_sentence_within_budget():
    question = "设备通过【1】连接洁净室"
    hit = SearchHit(
        "设备手册.md",
        "无关背景。" * 30 + "设备通过数据总线连接洁净室。" + "其他说明。" * 30,
        2.0,
    )

    snippets = extract_answerable_snippets(question, [hit], max_chars=80)

    assert len(snippets) == 1
    assert len(snippets[0].text) <= 80
    assert "数据总线" in snippets[0].text
