from __future__ import annotations

from pathlib import Path

from common.data.qa_coldstart import (
    build_coldstart_trajectories,
    expected_answer_payload,
    split_trajectories,
)
from common.environments.qa_search_core import LocalMarkdownIndex, QASearchRunner
from common.rewards.qa_reward import qa_rule_reward_fn
from nemo_rl_lab.config_resolve import resolve

REPO_ROOT = Path(__file__).resolve().parent.parent


def _runner(tmp_path):
    (tmp_path / "manual.md").write_text(
        "# 系统说明\n\n设备通过数据总线连接洁净室。\n\n离子注入系统由离子源和分析磁场组成。",
        encoding="utf-8",
    )
    return QASearchRunner(
        LocalMarkdownIndex(tmp_path, chunk_chars=160),
        qa_rule_reward_fn,
        top_k=3,
        answerability_rerank=True,
        query_expansion=True,
        evidence_reward_scale=1.0,
        max_result_chars=500,
    )


def test_expected_answer_payload_uses_first_alternative_per_item():
    assert expected_answer_payload("[fill] 数据总线/bus ||| 洁净室/cleanroom") == (
        "fill",
        "数据总线; 洁净室",
    )


def test_coldstart_builder_keeps_grounded_open_and_closed_replay(tmp_path):
    rows = [
        {
            "query": "下面是一道填空题。\n题目：设备通过【1】连接洁净室",
            "expected_answer": "[fill] 数据总线",
        },
        {
            "query": "下面是一道填空题。\n题目：完全不存在的答案是【1】",
            "expected_answer": "[fill] 神秘介质",
        },
        {
            "query": "下面是一道单选题。\n题目：请选择\nA.甲\nB.乙",
            "expected_answer": "[single] A",
        },
    ]

    trajectories, stats = build_coldstart_trajectories(
        rows,
        _runner(tmp_path),
        lambda query: "PROMPT:" + query,
        target_open=1,
        target_closed=1,
        max_open_scan=2,
        max_closed_scan=1,
        seed=1,
    )

    assert len(trajectories) == 2
    assert stats["selected_by_type"] == {"fill": 1, "single": 1}
    grounded = next(row for row in trajectories if row["category"] == "open")
    assert grounded["evidence_coverage"] == 1.0
    assert grounded["messages"][1]["content"].startswith("<search>")
    assert grounded["messages"][2]["role"] == "environment"
    assert "\\boxed{数据总线}" in grounded["messages"][-1]["content"]


def test_coldstart_builder_can_require_fully_grounded_open_answer(tmp_path):
    rows = [
        {
            "query": "下面是一道填空题。\n题目：设备通过【1】连接【2】",
            "expected_answer": "[fill] 数据总线 ||| 洁净室",
        }
    ]
    (tmp_path / "manual.md").write_text("设备通过数据总线运行。", encoding="utf-8")
    runner = QASearchRunner(
        LocalMarkdownIndex(tmp_path, chunk_chars=160),
        qa_rule_reward_fn,
        top_k=3,
        answerability_rerank=True,
        query_expansion=True,
        evidence_reward_scale=1.0,
        max_result_chars=500,
    )

    trajectories, stats = build_coldstart_trajectories(
        rows,
        runner,
        lambda query: "PROMPT:" + query,
        target_open=1,
        target_closed=0,
        max_open_scan=1,
        max_closed_scan=0,
        min_open_evidence_coverage=1.0,
        seed=1,
    )

    assert trajectories == []
    assert stats["trajectory_count"] == 0


def test_split_trajectories_never_crosses_group_key():
    trajectories = [
        {
            "category": "open" if index < 4 else "closed",
            "group_key": f"q-{index // 2}",
            "messages": [],
        }
        for index in range(8)
    ]

    train, validation = split_trajectories(
        trajectories,
        validation_fraction=0.5,
        seed=3,
    )

    train_keys = {row["group_key"] for row in train}
    validation_keys = {row["group_key"] for row in validation}
    assert train_keys.isdisjoint(validation_keys)
    assert train and validation


def test_coldstart_sft_config_has_remote_master_config_fields():
    config = resolve(
        REPO_ROOT
        / "experiments"
        / "sft_qwen3.5-9b_qa-agent-coldstart_v1"
        / "config.yaml"
    )

    assert config["sft"]["only_unmask_final"] is True
    assert config["data"]["data_build_only"] is True
    assert config["data"]["min_open_evidence_coverage"] == 0.0
    assert config["policy"]["tokenizer"]["chat_template"] is None
    assert config["policy"]["generation"]["colocated"]["enabled"] is False
    assert set(config["policy"]["generation"]) >= {
        "backend",
        "max_new_tokens",
        "temperature",
        "top_p",
        "top_k",
        "stop_token_ids",
        "stop_strings",
        "vllm_cfg",
    }
