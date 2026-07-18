#!/usr/bin/env python
"""Qwen 3.5 9B 的 QA 多轮检索 GRPO 入口。"""

from __future__ import annotations

import argparse
import json
import os
import pprint
import sys
from collections import Counter
from typing import Any

from omegaconf import OmegaConf
from torch.utils.data import Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir

from common.environments.qa_retrieval_eval import evaluate_retrieval_ab
from common.environments.qa_search_core import (
    STOP_STRINGS,
    LocalMarkdownIndex,
    qa_type_from_text,
)
from common.environments.qa_search_env import QASearchEnv

TASK_NAME = "qa_search"


def parse_args():
    parser = argparse.ArgumentParser(description="QA 多轮检索 GRPO")
    parser.add_argument("--config", type=str, default=None)
    args, overrides = parser.parse_known_args()
    return args, overrides


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class QAAgentDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        input_key: str,
        output_key: str,
        system_prompt: str,
        chat_template_kwargs: dict[str, Any] | None = None,
    ):
        self.rows = _read_jsonl(path)
        self.tokenizer = tokenizer
        self.input_key = input_key
        self.output_key = output_key
        self.system_prompt = system_prompt
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        type_counts = Counter(
            qa_type_from_text(str(row[self.input_key])) for row in self.rows
        )
        print(f"数据集 {path} 题型分布：{dict(sorted(type_counts.items()))}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> DatumSpec:
        row = self.rows[idx]
        query = str(row[self.input_key])
        expected = str(row[self.output_key])
        chat = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        prompt = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
            **self.chat_template_kwargs,
        ).strip()
        token_ids = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        message_log: LLMMessageLogType = [
            {"role": "user", "content": prompt, "token_ids": token_ids}
        ]
        return {
            "message_log": message_log,
            "length": len(token_ids),
            "extra_env_info": {
                "expected_answer": expected,
                "query": query,
                "searches": 0,
                "must_answer": False,
                "correction_used": False,
            },
            "loss_multiplier": 1.0,
            "idx": idx,
            "task_name": TASK_NAME,
            "stop_strings": list(STOP_STRINGS),
        }


def _run_retrieval_diagnostic(config: MasterConfig) -> None:
    data_cfg: dict[str, Any] = config.data
    data_dir = os.environ.get("QA_RL_DATA_DIR") or data_cfg.get("data_dir")
    if not data_dir:
        raise SystemExit("缺少 QA_RL_DATA_DIR 或 data.data_dir")

    env_cfg = dict(config.env[TASK_NAME]["cfg"])
    index = LocalMarkdownIndex(
        env_cfg.get("docs_dir", "/data/docs"),
        chunk_chars=int(env_cfg.get("chunk_chars", 480)),
        k1=float(env_cfg.get("k1", 1.5)),
        b=float(env_cfg.get("b", 0.75)),
        expand_ascii_tokens=bool(env_cfg.get("expand_ascii_tokens", False)),
    )
    train_rows = _read_jsonl(os.path.join(data_dir, "train.jsonl"))
    report = evaluate_retrieval_ab(
        train_rows,
        index,
        input_key=str(data_cfg.get("input_key", "query")),
        output_key=str(data_cfg.get("output_key", "expected_answer")),
        max_per_type=int(data_cfg.get("retrieval_diagnostic_max_per_type", 64)),
        seed=int(data_cfg.get("retrieval_diagnostic_seed", config.grpo["seed"])),
        top_k=int(env_cfg.get("top_k", 3)),
        candidate_k=int(env_cfg.get("candidate_k", 20)),
        query_expansion=bool(env_cfg.get("query_expansion", False)),
    )
    print(f"文档索引完成：{len(index.chunks)} 个片段，目录 {index.docs_dir}")
    print("QA检索A/B：" + json.dumps(report, ensure_ascii=False, sort_keys=True))


def main():
    register_omegaconf_resolvers()
    args, overrides = parse_args()
    if not args.config:
        args.config = os.path.join(THIS_DIR, "config.yaml")

    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    config = MasterConfig(**OmegaConf.to_container(config, resolve=True))
    pprint.pprint(config)

    if bool(config.data.get("retrieval_diagnostic", False)):
        _run_retrieval_diagnostic(config)
        return

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    init_ray()
    set_seed(config.grpo["seed"])

    tokenizer = get_tokenizer(config.policy["tokenizer"])
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"], tokenizer
    )

    data_cfg: dict[str, Any] = config.data
    data_dir = os.environ.get("QA_RL_DATA_DIR") or data_cfg.get("data_dir")
    if not data_dir:
        raise SystemExit("缺少 QA_RL_DATA_DIR 或 data.data_dir")
    system_prompt = str(data_cfg.get("system_prompt") or "").strip()
    if not system_prompt:
        raise SystemExit("data.system_prompt 不能为空")

    dataset_args = (
        tokenizer,
        str(data_cfg.get("input_key", "query")),
        str(data_cfg.get("output_key", "expected_answer")),
        system_prompt,
        dict(config.policy["tokenizer"].get("chat_template_kwargs") or {}),
    )
    train_dataset = QAAgentDataset(
        os.path.join(data_dir, "train.jsonl"),
        *dataset_args,
    )
    val_dataset = QAAgentDataset(os.path.join(data_dir, "val.jsonl"), *dataset_args)
    print(f"训练集 {len(train_dataset)} 条，验证集 {len(val_dataset)} 条")

    env_cfg = dict(config.env[TASK_NAME]["cfg"])
    env = QASearchEnv.options(num_gpus=0).remote(cfg=env_cfg)
    task_to_env = {TASK_NAME: env}

    (
        policy,
        policy_generation,
        _nemo_gym,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, train_dataset, val_dataset)

    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


if __name__ == "__main__":
    main()
