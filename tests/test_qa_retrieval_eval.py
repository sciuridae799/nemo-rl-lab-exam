"""答案证据检索 A/B 的本地测试。"""

from __future__ import annotations

import numpy as np
import pytest

from common.environments.qa_retrieval_eval import (
    _boxed_from_expected,
    _fit_linear_reranker,
    _linear_reranker_cross_validation,
    _predict_linear_reranker,
    evaluate_llm_candidate_reranker,
    evaluate_retrieval_ab,
    evaluate_supervised_query_expansion,
    evidence_coverage,
)
from common.environments.qa_search_core import LocalMarkdownIndex, SearchHit


def test_evidence_coverage_accepts_gold_alternatives():
    hits = [
        SearchHit(
            "manual.md",
            "SERVER ROOM 通过 SQL Server 与 Clean room 连接。",
            1.0,
        )
    ]
    assert evidence_coverage("[fill] SQL server/SQL服务器", hits) == 1.0
    assert evidence_coverage("[fill] Air shower", hits) == 0.0


def test_boxed_from_expected_formats_short_answer_items():
    assert _boxed_from_expected("[short] 要点一 ||| 要点二") == "\\boxed{要点一 ; 要点二}"


def test_llm_candidate_reranker_report_uses_selected_ids(tmp_path):
    (tmp_path / "模块-试卷.md").write_text(
        "# 试卷\n\n设备通过【1】连接系统。",
        encoding="utf-8",
    )
    (tmp_path / "模块-答案.md").write_text(
        "# 参考答案\n\n设备通过数据总线连接系统。",
        encoding="utf-8",
    )
    index = LocalMarkdownIndex(tmp_path, chunk_chars=160)

    class FirstCandidateReranker:
        def select(self, question, candidates, *, limit=3):
            assert question and candidates and limit == 3
            return [0], "[1]"

    report = evaluate_llm_candidate_reranker(
        [
            {
                "query": "下面是一道填空题。\n\n题目：设备通过【1】连接系统",
                "expected_answer": "[fill] 数据总线",
            }
        ],
        index,
        FirstCandidateReranker(),
        max_per_type=1,
        candidate_k=4,
        candidate_max_per_source=2,
        shortlist_size=4,
    )

    assert report["sample_count"] == 1
    assert report["mean_selected_count"] == 1.0


def test_retrieval_ab_reports_answer_evidence_gain(tmp_path):
    question = "SERVER ROOM 通过【1】与Clean room进行连接"
    for index in range(6):
        (tmp_path / f"exam-{index}.md").write_text(
            f"# 试卷 {index}\n\n" + ("1. 填空题（每题4分）\n3. SERVER ROOM 通过 与Clean room进行连接\n\n") * 4,
            encoding="utf-8",
        )
    (tmp_path / "manual.md").write_text(
        "# Server 架构\n\nSERVER ROOM 通过 SQL server 与 Clean room 进行连接。",
        encoding="utf-8",
    )
    index = LocalMarkdownIndex(tmp_path, chunk_chars=200)
    rows = [
        {
            "query": f"下面是一道填空题。\n\n题目：{question}",
            "expected_answer": "[fill] SQL server",
        }
    ]

    report = evaluate_retrieval_ab(
        rows,
        index,
        max_per_type=1,
        top_k=3,
        candidate_k=10,
    )

    assert report["sample_count"] == 1
    assert report["baseline"]["evidence_coverage"] == pytest.approx(0.0)
    assert report["reranked"]["evidence_coverage"] == pytest.approx(1.0)
    assert report["delta"]["evidence_coverage"] == pytest.approx(1.0)
    assert report["reranked"]["question_copy_rate"] < report["baseline"]["question_copy_rate"]
    assert report["funnel"]["corpus_evidence_coverage"] == pytest.approx(1.0)
    assert report["funnel"]["candidate_pool_evidence_coverage"] == pytest.approx(1.0)
    assert report["funnel"]["reranked_top3_evidence_coverage"] == pytest.approx(1.0)
    assert report["funnel"]["packed_top8_evidence_coverage"] == pytest.approx(1.0)


def test_retrieval_ab_can_expand_sibling_answer_document(tmp_path):
    question = "设备通过【1】连接洁净室"
    (tmp_path / "模块-试卷.md").write_text(
        "# 试卷\n\n" + (f"填空题：{question}\n" * 6),
        encoding="utf-8",
    )
    (tmp_path / "模块-答案.md").write_text(
        "# 参考答案\n\n设备通过数据总线连接洁净室。",
        encoding="utf-8",
    )
    index = LocalMarkdownIndex(tmp_path, chunk_chars=160)
    rows = [
        {
            "query": f"下面是一道填空题。\n\n题目：{question}",
            "expected_answer": "[fill] 数据总线",
        }
    ]

    report = evaluate_retrieval_ab(
        rows,
        index,
        max_per_type=1,
        top_k=1,
        candidate_k=1,
        structural_expansion=True,
    )

    assert report["structural_expansion"] is True
    assert report["baseline"]["evidence_coverage"] == 0.0
    assert report["reranked"]["evidence_coverage"] == 1.0
    assert report["funnel"]["candidate_pool_evidence_coverage"] == 0.0
    assert report["funnel"]["expanded_pool_evidence_coverage"] == 1.0
    assert report["funnel"]["structural_coverage_improved_samples"] == 1


def test_supervised_query_expansion_uses_train_only_neighbors(tmp_path):
    pytest.importorskip("sklearn")
    (tmp_path / "manual.md").write_text(
        "# Manual\n\n"
        "设备甲通过数据总线连接洁净室。\n"
        "设备乙通过光纤连接洁净室。\n"
        "设备丙通过控制线连接洁净室。\n",
        encoding="utf-8",
    )
    index = LocalMarkdownIndex(tmp_path, chunk_chars=200)
    rows = [
        {
            "query": f"下面是一道填空题。\n\n题目：设备{label}通过【1】连接洁净室",
            "expected_answer": f"[fill] {answer}",
        }
        for label, answer in (
            ("甲", "数据总线"),
            ("乙", "光纤"),
            ("丙", "控制线"),
            ("丁", "数据总线"),
            ("戊", "光纤"),
            ("己", "控制线"),
            ("庚", "数据总线"),
            ("辛", "光纤"),
        )
    ]
    report = evaluate_supervised_query_expansion(
        rows,
        index,
        candidate_k=6,
        candidate_max_per_source=3,
        structural_expansion=False,
        aligned_sibling_expansion=False,
        min_similarity=0.0,
    )

    assert report["sample_count"] > 0
    assert report["train_count"] > 0
    assert "delta" in report


def test_retrieval_funnel_separates_corpus_presence_from_query_recall(tmp_path):
    (tmp_path / "unrelated.md").write_text(
        "# 参考资料\n\n系统最终使用数据总线。",
        encoding="utf-8",
    )
    index = LocalMarkdownIndex(tmp_path, chunk_chars=160)
    rows = [
        {
            "query": "下面是一道填空题。\n\n题目：完全无关的提问【1】",
            "expected_answer": "[fill] 数据总线",
        }
    ]

    report = evaluate_retrieval_ab(
        rows,
        index,
        max_per_type=1,
        top_k=1,
        candidate_k=1,
    )

    assert report["funnel"]["corpus_evidence_coverage"] == 1.0
    assert report["funnel"]["candidate_pool_evidence_coverage"] == 0.0
    assert report["funnel"]["candidate_missed_corpus_samples"] == 1


def test_linear_reranker_fit_prefers_positive_feature():
    groups = [
        {
            "features": [[0.0, 1.0], [1.0, 0.0]],
            "labels": [0.0, 1.0],
        },
        {
            "features": [[0.2, 1.0], [0.8, 0.0]],
            "labels": [0.0, 1.0],
        },
    ]

    model = _fit_linear_reranker(groups, l2=0.01)
    predictions = _predict_linear_reranker(
        model,
        np.asarray([[0.1, 1.0], [0.9, 0.0]]),
    )

    assert predictions[1] > predictions[0]


def test_linear_reranker_cross_validation_holds_out_whole_questions():
    groups = []
    for index in range(8):
        positive = [0.0] * 11
        positive[0] = 1.0
        negative = [0.0] * 11
        groups.append({
            "type": "fill" if index % 2 == 0 else "short",
            "group_key": f"question-{index // 4}",
            "expected": "[fill] 数据总线",
            "hits": [
                SearchHit(f"answer-{index}.md", "数据总线", 2.0),
                SearchHit(f"exam-{index}.md", "空白题面", 1.0),
            ],
            "features": np.asarray([positive, negative]),
            "labels": np.asarray([1.0, 0.0]),
            "baseline": 0.0,
        })

    report = _linear_reranker_cross_validation(groups, top_k=1, folds=4)

    assert report["sample_count"] == 8
    assert report["unique_question_count"] == 4
    assert report["duplicate_question_rows"] == 4
    assert report["heldout_evidence_coverage"] == 1.0
    assert report["by_type"]["fill"]["count"] == 4
