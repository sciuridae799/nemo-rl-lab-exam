#!/usr/bin/env python3
"""构造训练集正工具轨迹，并从 F4 模型权重做短 SFT。"""

from __future__ import annotations

import argparse
import json
import os
import pprint
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
NEMO_RL_DIR = Path(os.environ.get("NEMO_RL_DIR", ""))
if NEMO_RL_DIR.is_dir():
    sys.path.insert(0, str(NEMO_RL_DIR))

import torch
from examples.run_sft import setup_data
from nemo_rl.algorithms.sft import (
    MasterConfig,
    setup,
    sft_train,
    validate,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.utils.checkpoint import CheckpointManager
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from common.data.qa_coldstart import (
    build_coldstart_trajectories,
    split_trajectories,
)
from common.environments.qa_search_core import LocalMarkdownIndex, QASearchRunner
from common.rewards.qa_reward import qa_rule_reward_fn
from common.utils.checkpoint_seed import seed_checkpoint_step


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_known_args()


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_openai_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False))
            handle.write("\n")


def _render_prompt_factory(model_name: str, system_prompt: str):
    raw_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    def render(query: str) -> str:
        return raw_tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
            enable_thinking=False,
        ).strip()

    return render


def _build_runner(config: dict[str, Any]) -> QASearchRunner:
    retrieval = dict(config["retrieval"])
    index = LocalMarkdownIndex(
        retrieval["docs_dir"],
        chunk_chars=int(retrieval["chunk_chars"]),
        k1=float(retrieval["k1"]),
        b=float(retrieval["b"]),
    )
    print(f"冷启动文档索引：{len(index.chunks)} 个片段")
    return QASearchRunner(
        index,
        qa_rule_reward_fn,
        top_k=int(retrieval["top_k"]),
        candidate_k=int(retrieval["candidate_k"]),
        candidate_max_per_source=int(retrieval["candidate_max_per_source"]),
        answerability_rerank=bool(retrieval["answerability_rerank"]),
        query_expansion=bool(retrieval["query_expansion"]),
        structural_expansion=bool(retrieval["structural_expansion"]),
        max_searches=int(retrieval["max_searches"]),
        max_result_chars=int(retrieval["max_result_chars"]),
        # 只借用 metadata 计算证据覆盖，搜索 observation 和轮奖励不变。
        evidence_reward_scale=1.0,
    )


def _seed_f4_weights_only(
    config: dict[str, Any],
    train_dataset,
) -> Path:
    data_cfg = config["data"]
    source_root = str(data_cfg.get("resume_checkpoint_dir") or "").strip()
    if not source_root:
        raise SystemExit("缺少 data.resume_checkpoint_dir")
    source_step = int(data_cfg.get("resume_checkpoint_step", 30))
    target_root = Path(config["checkpointing"]["checkpoint_dir"])
    target_step = seed_checkpoint_step(
        source_root,
        target_root,
        source_step,
        target_step=0,
    )

    training_info_path = target_step / "training_info.json"
    training_info_path.unlink()
    training_info_path.write_text(
        json.dumps(
            {
                "epoch": 0,
                "step": 0,
                "total_steps": 0,
                "consumed_samples": 0,
                "total_valid_tokens": 0,
            }
        ),
        encoding="utf-8",
    )

    fresh_loader = StatefulDataLoader(
        train_dataset,
        batch_size=int(config["policy"]["train_global_batch_size"]),
        shuffle=bool(config["data"]["shuffle"]),
        collate_fn=rl_collate_fn,
        drop_last=True,
        num_workers=int(config["data"]["num_workers"]),
    )
    dataloader_path = target_step / "train_dataloader.pt"
    dataloader_path.unlink()
    torch.save(fresh_loader.state_dict(), dataloader_path)

    original_get_resume_paths = CheckpointManager.get_resume_paths

    def weights_only(last_checkpoint_path):
        weights_path, _ = original_get_resume_paths(last_checkpoint_path)
        return weights_path, None

    CheckpointManager.get_resume_paths = staticmethod(weights_only)
    print(
        f"F4 weights-only 播种完成：{source_root}/step_{source_step} -> {target_step}；"
        "SFT 状态和 dataloader 已重置，optimizer/scheduler 强制新建"
    )
    return target_step


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = parse_args()
    if not args.config:
        args.config = str(Path(__file__).with_name("config.yaml"))
    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    config = OmegaConf.to_container(config, resolve=True)
    assert isinstance(config, dict)
    pprint.pprint(config)

    data_cfg = config["data"]
    data_dir = Path(os.environ.get("QA_RL_DATA_DIR") or data_cfg["data_dir"])
    train_rows = _read_jsonl(data_dir / "train.jsonl")
    system_prompt = str(data_cfg["system_prompt"]).strip()
    runner = _build_runner(config)
    render_prompt = _render_prompt_factory(config["policy"]["model_name"], system_prompt)
    trajectories, build_stats = build_coldstart_trajectories(
        train_rows,
        runner,
        render_prompt,
        target_open=int(data_cfg["target_open"]),
        target_closed=int(data_cfg["target_closed"]),
        max_open_scan=int(data_cfg["max_open_scan_per_type"]),
        max_closed_scan=int(data_cfg["max_closed_scan_per_type"]),
        seed=int(data_cfg["seed"]),
    )
    train_trajectories, validation_trajectories = split_trajectories(
        trajectories,
        validation_fraction=float(data_cfg["validation_fraction"]),
        seed=int(data_cfg["seed"]),
    )
    build_stats.update({
        "train_trajectory_count": len(train_trajectories),
        "validation_trajectory_count": len(validation_trajectories),
    })
    print("QA冷启动数据：" + json.dumps(build_stats, ensure_ascii=False, sort_keys=True))
    open_count = sum(row["category"] == "open" for row in trajectories)
    if open_count < int(data_cfg["min_open_trajectories"]):
        raise RuntimeError(
            f"开放题正轨迹仅 {open_count}，低于门槛 {data_cfg['min_open_trajectories']}"
        )
    if len(train_trajectories) < int(config["policy"]["train_global_batch_size"]):
        raise RuntimeError("训练轨迹不足一个 global batch")
    if not validation_trajectories:
        raise RuntimeError("冷启动 holdout 为空")

    output_root = Path(config["checkpointing"]["checkpoint_dir"])
    data_output = output_root / "coldstart_data"
    train_path = data_output / "train.jsonl"
    validation_path = data_output / "validation.jsonl"
    _write_openai_jsonl(train_path, train_trajectories)
    _write_openai_jsonl(validation_path, validation_trajectories)
    common_dataset_cfg = {
        "dataset_name": "openai_format",
        "chat_key": "messages",
        "tool_key": None,
        "use_preserving_dataset": False,
    }
    config["data"]["train"] = {**common_dataset_cfg, "data_path": str(train_path)}
    config["data"]["validation"] = {
        **common_dataset_cfg,
        "data_path": str(validation_path),
    }

    config["logger"]["log_dir"] = get_next_experiment_dir(
        config["logger"]["log_dir"]
    )
    init_ray()
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    train_dataset, val_dataset = setup_data(tokenizer, config["data"])
    _seed_f4_weights_only(config, train_dataset)

    master_config: MasterConfig = config
    (
        policy,
        _cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        sft_save_state,
        master_config,
    ) = setup(master_config, tokenizer, train_dataset, val_dataset)

    if bool(data_cfg["weights_load_probe_only"]):
        val_metrics, validation_timings = validate(
            policy,
            val_dataloader,
            tokenizer,
            loss_fn,
            step=0,
            master_config=master_config,
            val_batches=int(config["sft"]["val_batches"]),
            val_batch_size=int(config["sft"]["val_global_batch_size"]),
            val_mbs=int(config["sft"]["val_micro_batch_size"]),
        )
        logger.log_metrics(val_metrics, 0, prefix="validation")
        logger.log_metrics(validation_timings, 0, prefix="timing/validation")
        print("F4 weights-only SFT 探针：" + json.dumps(val_metrics, sort_keys=True))
        return

    sft_train(
        policy,
        train_dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        master_config,
        logger,
        checkpointer,
        sft_save_state,
    )


if __name__ == "__main__":
    main()

