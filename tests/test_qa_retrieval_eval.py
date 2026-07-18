"""答案证据检索 A/B 的本地测试。"""

from __future__ import annotations

import pytest

from common.environments.qa_retrieval_eval import (
    evaluate_retrieval_ab,
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
