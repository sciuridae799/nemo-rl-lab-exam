#!/usr/bin/env python
"""Qwen 3.5 9B 的 QA 多轮检索 GRPO 入口。"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import pprint
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from nemo_rl.algorithms.grpo import (
    MasterConfig,
    grpo_train,
    refit_policy_generation,
    setup,
    validate,
)
from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.checkpoint import CheckpointManager
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir

from common.environments.qa_llm_reranker import QwenCandidateReranker
from common.environments.qa_retrieval_eval import (
    evaluate_answerability_weight_grid,
    evaluate_llm_candidate_reranker,
    evaluate_llm_query_rewrite,
    evaluate_llm_teacher_agent,
    evaluate_qa_memory_knn,
    evaluate_qa_memory_option_mapping,
    evaluate_retrieval_ab,
    evaluate_supervised_query_expansion,
)
from common.environments.qa_search_core import (
    STOP_STRINGS,
    LocalMarkdownIndex,
    filter_qa_rows_by_type,
    qa_loss_multiplier,
    qa_type_from_text,
)
from common.environments.qa_search_env import QASearchEnv
from common.utils.checkpoint_seed import seed_checkpoint_step

TASK_NAME = "qa_search"


def _runtime_capability_probe() -> None:
    """只读检查远端已有语义检索能力，不初始化模型或索引。"""
    package_names = (
        "sentence-transformers",
        "transformers",
        "faiss-cpu",
        "faiss-gpu",
        "scikit-learn",
        "numpy",
    )
    module_names = (
        "sentence_transformers",
        "transformers",
        "faiss",
        "sklearn",
        "numpy",
    )
    packages: dict[str, str | None] = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    modules = {
        name: importlib.util.find_spec(name) is not None for name in module_names
    }
    model_configs: list[dict[str, Any]] = []
    for root_name in ("/data/huggingface", "/data/models"):
        root = Path(root_name)
        if not root.is_dir():
            continue
        for config_path in sorted(root.rglob("config.json")):
            try:
                relative = config_path.relative_to(root)
                if len(relative.parts) > 7:
                    continue
                config_data = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
                continue
            model_type = config_data.get("model_type")
            architectures = config_data.get("architectures")
            if not model_type and not architectures:
                continue
            model_configs.append(
                {
                    "path": str(config_path.parent),
                    "model_type": model_type,
                    "architectures": architectures,
                }
            )
            if len(model_configs) >= 200:
                break
        if len(model_configs) >= 200:
            break

    print(
        "QA远端能力探针："
        + json.dumps(
            {
                "packages": packages,
                "modules": modules,
                "model_configs": model_configs,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


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
        open_loss_multiplier: float = 1.0,
        *,
        is_training: bool,
        allowed_question_types: tuple[str, ...] | None = None,
    ):
        self.is_training = bool(is_training)
        self.rows = _read_jsonl(path)
        if self.is_training and allowed_question_types:
            original_count = len(self.rows)
            self.rows = filter_qa_rows_by_type(
                self.rows,
                allowed_question_types,
                input_key=input_key,
            )
            print(
                f"训练集题型过滤：{original_count} -> {len(self.rows)} 条；"
                f"允许 {sorted(set(allowed_question_types))}"
            )
        self.tokenizer = tokenizer
        self.input_key = input_key
        self.output_key = output_key
        self.system_prompt = system_prompt
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.open_loss_multiplier = float(open_loss_multiplier)
        type_counts = Counter(
            qa_type_from_text(str(row[self.input_key])) for row in self.rows
        )
        print(
            f"数据集 {path} 题型分布：{dict(sorted(type_counts.items()))}；"
            f"开放题 loss multiplier={(self.open_loss_multiplier if self.is_training else 1.0):.3f}"
        )

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
                # 环境仅据此决定是否启用训练 shaping；验证始终保持官方奖励。
                "is_training": self.is_training,
                "evidence_coverage": 0.0,
                "evidence_reward_total": 0.0,
            },
            "loss_multiplier": qa_loss_multiplier(
                query,
                is_training=self.is_training,
                open_loss_multiplier=self.open_loss_multiplier,
            ),
            "idx": idx,
            "task_name": TASK_NAME,
            "stop_strings": list(STOP_STRINGS),
        }


def _run_retrieval_diagnostic(config: MasterConfig) -> None:
    data_cfg: dict[str, Any] = config.data
    data_dir = os.environ.get("QA_RL_DATA_DIR") or data_cfg.get("data_dir")
    if not data_dir:
        raise SystemExit("缺少 QA_RL_DATA_DIR 或 data.data_dir")

    train_rows = _read_jsonl(os.path.join(data_dir, "train.jsonl"))
    if bool(data_cfg.get("qa_memory_mapping_diagnostic", False)):
        report = evaluate_qa_memory_option_mapping(
            train_rows,
            input_key=str(data_cfg.get("input_key", "query")),
            output_key=str(data_cfg.get("output_key", "expected_answer")),
        )
        print("QA记忆选项映射门控：" + json.dumps(report, ensure_ascii=False, sort_keys=True))
        return
    env_cfg = dict(config.env[TASK_NAME]["cfg"])
    index = LocalMarkdownIndex(
        env_cfg.get("docs_dir", "/data/docs"),
        chunk_chars=int(env_cfg.get("chunk_chars", 480)),
        k1=float(env_cfg.get("k1", 1.5)),
        b=float(env_cfg.get("b", 0.75)),
        expand_ascii_tokens=bool(env_cfg.get("expand_ascii_tokens", False)),
    )
    if bool(data_cfg.get("teacher_validation_diagnostic", False)):
        model_path = str(data_cfg.get("llm_reranker_model_path") or "").strip()
        if not model_path:
            raise SystemExit("缺少 data.llm_reranker_model_path")
        val_rows = _read_jsonl(os.path.join(data_dir, "val.jsonl"))
        teacher = QwenCandidateReranker(model_path)
        report = evaluate_llm_teacher_agent(
            val_rows,
            index,
            teacher,
            input_key=str(data_cfg.get("input_key", "query")),
            output_key=str(data_cfg.get("output_key", "expected_answer")),
            max_per_type=int(data_cfg.get("teacher_diagnostic_max_per_type", 4)),
            seed=int(data_cfg.get("retrieval_diagnostic_seed", config.grpo["seed"])),
            top_k=int(env_cfg.get("top_k", 3)),
            candidate_k=int(env_cfg.get("candidate_k", 80)),
            candidate_max_per_source=int(env_cfg.get("candidate_max_per_source", 4)),
            query_expansion=bool(env_cfg.get("query_expansion", True)),
            structural_expansion=bool(env_cfg.get("structural_expansion", False)),
            aligned_sibling_expansion=bool(env_cfg.get("aligned_sibling_expansion", False)),
            max_searches=int(env_cfg.get("max_searches", 2)),
            max_result_chars=int(env_cfg.get("max_result_chars", 1200)),
        )
        print(f"文档索引完成：{len(index.chunks)} 个片段，目录 {index.docs_dir}")
        print("LLM教师端到端门控：" + json.dumps(report, ensure_ascii=False, sort_keys=True))
        return
    if bool(data_cfg.get("llm_query_rewrite_diagnostic", False)):
        model_path = str(data_cfg.get("llm_reranker_model_path") or "").strip()
        if not model_path:
            raise SystemExit("缺少 data.llm_reranker_model_path")
        reranker = QwenCandidateReranker(model_path)
        report = evaluate_llm_query_rewrite(
            train_rows,
            index,
            reranker,
            input_key=str(data_cfg.get("input_key", "query")),
            output_key=str(data_cfg.get("output_key", "expected_answer")),
            max_per_type=int(data_cfg.get("retrieval_diagnostic_max_per_type", 8)),
            seed=int(data_cfg.get("retrieval_diagnostic_seed", config.grpo["seed"])),
            top_k=int(env_cfg.get("top_k", 3)),
            candidate_k=int(env_cfg.get("candidate_k", 80)),
            candidate_max_per_source=int(env_cfg.get("candidate_max_per_source", 4)),
        )
        print(f"文档索引完成：{len(index.chunks)} 个片段，目录 {index.docs_dir}")
        print("LLM查询改写门控：" + json.dumps(report, ensure_ascii=False, sort_keys=True))
        return
    if bool(data_cfg.get("llm_reranker_diagnostic", False)):
        model_path = str(data_cfg.get("llm_reranker_model_path") or "").strip()
        if not model_path:
            raise SystemExit("缺少 data.llm_reranker_model_path")
        reranker = QwenCandidateReranker(model_path)
        report = evaluate_llm_candidate_reranker(
            train_rows,
            index,
            reranker,
            input_key=str(data_cfg.get("input_key", "query")),
            output_key=str(data_cfg.get("output_key", "expected_answer")),
            max_per_type=int(data_cfg.get("retrieval_diagnostic_max_per_type", 16)),
            seed=int(data_cfg.get("retrieval_diagnostic_seed", config.grpo["seed"])),
            top_k=int(env_cfg.get("top_k", 3)),
            candidate_k=int(env_cfg.get("candidate_k", 80)),
            candidate_max_per_source=int(env_cfg.get("candidate_max_per_source", 4)),
        )
        print(f"文档索引完成：{len(index.chunks)} 个片段，目录 {index.docs_dir}")
        print("LLM语义重排门控：" + json.dumps(report, ensure_ascii=False, sort_keys=True))
        return
    if bool(data_cfg.get("weight_grid_diagnostic", False)):
        report = evaluate_answerability_weight_grid(
            train_rows,
            index,
            input_key=str(data_cfg.get("input_key", "query")),
            output_key=str(data_cfg.get("output_key", "expected_answer")),
            max_per_type=int(data_cfg.get("retrieval_diagnostic_max_per_type", 32)),
            seed=int(data_cfg.get("retrieval_diagnostic_seed", config.grpo["seed"])),
            top_k=int(env_cfg.get("top_k", 3)),
            candidate_k=int(env_cfg.get("candidate_k", 80)),
            candidate_max_per_source=int(env_cfg.get("candidate_max_per_source", 4)),
            query_expansion=bool(env_cfg.get("query_expansion", True)),
            structural_expansion=bool(env_cfg.get("structural_expansion", False)),
            aligned_sibling_expansion=bool(env_cfg.get("aligned_sibling_expansion", False)),
        )
        print(f"文档索引完成：{len(index.chunks)} 个片段，目录 {index.docs_dir}")
        print("可回答性权重网格：" + json.dumps(report, ensure_ascii=False, sort_keys=True))
        return
    if bool(data_cfg.get("supervised_query_diagnostic", False)):
        report = evaluate_supervised_query_expansion(
            train_rows,
            index,
            input_key=str(data_cfg.get("input_key", "query")),
            output_key=str(data_cfg.get("output_key", "expected_answer")),
            top_k=int(env_cfg.get("top_k", 3)),
            candidate_k=int(env_cfg.get("candidate_k", 80)),
            candidate_max_per_source=int(env_cfg.get("candidate_max_per_source", 4)),
            query_expansion=bool(env_cfg.get("query_expansion", True)),
            structural_expansion=bool(env_cfg.get("structural_expansion", False)),
            aligned_sibling_expansion=bool(env_cfg.get("aligned_sibling_expansion", False)),
        )
        print(f"文档索引完成：{len(index.chunks)} 个片段，目录 {index.docs_dir}")
        print("监督查询扩展门控：" + json.dumps(report, ensure_ascii=False, sort_keys=True))
        return
    report = evaluate_retrieval_ab(
        train_rows,
        index,
        input_key=str(data_cfg.get("input_key", "query")),
        output_key=str(data_cfg.get("output_key", "expected_answer")),
        max_per_type=int(data_cfg.get("retrieval_diagnostic_max_per_type", 64)),
        seed=int(data_cfg.get("retrieval_diagnostic_seed", config.grpo["seed"])),
        top_k=int(env_cfg.get("top_k", 3)),
        candidate_k=int(env_cfg.get("candidate_k", 20)),
        candidate_max_per_source=int(env_cfg.get("candidate_max_per_source", 4)),
        query_expansion=bool(env_cfg.get("query_expansion", False)),
        structural_expansion=bool(env_cfg.get("structural_expansion", False)),
        aligned_sibling_expansion=bool(env_cfg.get("aligned_sibling_expansion", False)),
        packing_top_k=int(env_cfg.get("packing_top_k", 8)),
        packing_snippet_chars=int(env_cfg.get("packing_snippet_chars", 140)),
    )
    print(f"文档索引完成：{len(index.chunks)} 个片段，目录 {index.docs_dir}")
    print("QA检索A/B：" + json.dumps(report, ensure_ascii=False, sort_keys=True))


def _run_qa_memory_diagnostic(config: MasterConfig) -> None:
    """不构建文档索引，只评估训练集问题记忆的分组 held-out 上限。"""
    data_cfg: dict[str, Any] = config.data
    data_dir = os.environ.get("QA_RL_DATA_DIR") or data_cfg.get("data_dir")
    if not data_dir:
        raise SystemExit("缺少 QA_RL_DATA_DIR 或 data.data_dir")
    train_rows = _read_jsonl(os.path.join(data_dir, "train.jsonl"))
    report = evaluate_qa_memory_knn(
        train_rows,
        input_key=str(data_cfg.get("input_key", "query")),
        output_key=str(data_cfg.get("output_key", "expected_answer")),
    )
    print("QA训练记忆门控：" + json.dumps(report, ensure_ascii=False, sort_keys=True))


def _run_llm_reranker_load_probe(config: MasterConfig) -> None:
    """加载平台已有指令模型并完成一个不含真实题目的重排冒烟。"""
    model_path = str(config.data.get("llm_reranker_model_path") or "").strip()
    if not model_path:
        raise SystemExit("缺少 data.llm_reranker_model_path")
    reranker = QwenCandidateReranker(model_path)
    selected, raw = reranker.select(
        "设备通过什么方式连接系统？",
        [
            ("示例手册", "设备通过数据总线连接系统。"),
            ("无关资料", "本页介绍人员培训安排。"),
        ],
    )
    print(
        "LLM重排加载探针："
        + json.dumps(
            {"selected": selected, "raw": raw, "model_path": model_path},
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _seed_grpo_weights_only(
    config: MasterConfig,
    train_dataset: QAAgentDataset,
    source_root: str,
    source_step: int,
) -> Path:
    """把跨算法 checkpoint 只作为 GRPO step 0 权重，重建全部训练状态。"""
    if bool(config.data.get("use_multiple_dataloader", False)):
        raise RuntimeError("weights-only 播种暂不支持 multiple dataloader")
    if bool(config.grpo.get("use_dynamic_sampling", False)):
        raise RuntimeError("weights-only 播种探针要求关闭 dynamic sampling")
    if int(config.grpo.get("batch_multiplier", 1)) != 1:
        raise RuntimeError("weights-only 播种要求 batch_multiplier=1")

    target_step = seed_checkpoint_step(
        source_root,
        str(config.checkpointing["checkpoint_dir"]),
        source_step,
        target_step=0,
    )
    training_info_path = target_step / "training_info.json"
    training_info_path.unlink()
    training_info_path.write_text(
        json.dumps(
            {
                "consumed_samples": 0,
                "current_step": 0,
                "current_epoch": 0,
                "total_steps": 0,
                "total_valid_tokens": 0,
                "val_reward": -99999999.0,
            }
        ),
        encoding="utf-8",
    )

    fresh_loader = StatefulDataLoader(
        train_dataset,
        batch_size=int(config.grpo["num_prompts_per_step"]),
        shuffle=bool(config.data["shuffle"]),
        collate_fn=rl_collate_fn,
        drop_last=True,
        num_workers=int(config.data["num_workers"]),
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
        f"GRPO weights-only 播种完成：{source_root}/step_{source_step} -> "
        f"{target_step}；GRPO state/dataloader 已重置，optimizer/scheduler 强制新建"
    )
    return target_step


def main():
    register_omegaconf_resolvers()
    args, overrides = parse_args()
    if not args.config:
        args.config = os.path.join(THIS_DIR, "config.yaml")

    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)

    resume_source = str(config.data.get("resume_checkpoint_dir") or "").strip()
    resume_step = int(config.data.get("resume_checkpoint_step", 0))
    resume_validation_only = bool(config.data.get("resume_validation_only", False))
    resume_weights_only = bool(config.data.get("resume_weights_only", False))
    weights_only_validation_only = bool(
        config.data.get("weights_only_validation_only", False)
    )
    if resume_validation_only and resume_weights_only:
        raise SystemExit("resume_validation_only 与 resume_weights_only 不能同时启用")
    if weights_only_validation_only and not resume_weights_only:
        raise SystemExit("weights_only_validation_only 需要 resume_weights_only=true")
    if resume_source and not resume_weights_only:
        seeded = seed_checkpoint_step(
            resume_source,
            str(config.checkpointing["checkpoint_dir"]),
            resume_step,
        )
        print(f"断点续训播种完成：{seeded}（历史目录保持只读）")
    elif resume_validation_only:
        raise SystemExit("resume_validation_only=true 时必须配置 resume_checkpoint_dir")

    config = MasterConfig(**OmegaConf.to_container(config, resolve=True))
    pprint.pprint(config)

    if bool(config.data.get("runtime_capability_probe", False)):
        _runtime_capability_probe()
        return

    if bool(config.data.get("llm_reranker_load_probe", False)):
        _run_llm_reranker_load_probe(config)
        return

    if bool(config.data.get("qa_memory_diagnostic", False)):
        _run_qa_memory_diagnostic(config)
        return

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
        float(data_cfg.get("open_loss_multiplier", 1.0)),
    )
    train_question_types = tuple(
        str(value) for value in (data_cfg.get("train_question_types") or [])
    )
    train_dataset = QAAgentDataset(
        os.path.join(data_dir, "train.jsonl"),
        *dataset_args,
        is_training=True,
        allowed_question_types=train_question_types,
    )
    val_dataset = QAAgentDataset(
        os.path.join(data_dir, "val.jsonl"),
        *dataset_args,
        is_training=False,
    )
    print(f"训练集 {len(train_dataset)} 条，验证集 {len(val_dataset)} 条")

    if resume_source and resume_weights_only:
        _seed_grpo_weights_only(
            config,
            train_dataset,
            resume_source,
            resume_step,
        )

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

    if resume_validation_only or weights_only_validation_only:
        current_step = int(grpo_state["current_step"])
        expected_step = 0 if weights_only_validation_only else resume_step
        if current_step != expected_step:
            raise RuntimeError(
                f"断点 step 校验失败：期望 {expected_step}，实际 {current_step}"
            )
        if policy_generation is None:
            raise RuntimeError("validation-only 探针需要独立生成后端")
        mode = "weights-only" if weights_only_validation_only else "resume"
        print(f"开始验证 {mode} policy：step={current_step}")
        refit_policy_generation(
            policy,
            policy_generation,
            bool(config.policy["generation"]["colocated"]["enabled"]),
        )
        val_metrics, validation_timings = validate(
            policy_generation,
            val_dataloader,
            tokenizer,
            task_to_env,
            step=current_step,
            master_config=master_config,
            logger=logger,
        )
        policy_generation.finish_generation()
        logger.log_metrics(val_metrics, current_step, prefix="validation")
        logger.log_metrics(
            validation_timings,
            current_step,
            prefix="timing/validation",
        )
        print(f"{mode} 验证指标：{dict(sorted(val_metrics.items()))}")
        return

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
