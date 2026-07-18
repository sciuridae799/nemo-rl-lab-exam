"""多轮本地文档检索 QA 环境。"""

from __future__ import annotations

from typing import Any

import ray
import torch
from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

from common.environments.qa_search_core import (
    LocalMarkdownIndex,
    QASearchRunner,
    qa_reward_diagnostics,
    qa_type_from_text,
)


@ray.remote  # pragma: no cover
class QASearchEnv(EnvironmentInterface[dict[str, Any]]):
    def __init__(self, cfg: dict[str, Any] | None = None):
        cfg = cfg or {}
        index = LocalMarkdownIndex(
            cfg.get("docs_dir", "/data/docs"),
            chunk_chars=int(cfg.get("chunk_chars", 480)),
            k1=float(cfg.get("k1", 1.5)),
            b=float(cfg.get("b", 0.75)),
            expand_ascii_tokens=bool(cfg.get("expand_ascii_tokens", False)),
        )
        if bool(cfg.get("use_judge", True)):
            from common.rewards.qa_judge_reward import (
                get_judge_stats,
                qa_judge_reward_fn,
            )

            reward_fn = qa_judge_reward_fn
            self._get_judge_stats = get_judge_stats
        else:
            from common.rewards.qa_reward import qa_rule_reward_fn

            reward_fn = qa_rule_reward_fn
            self._get_judge_stats = None

        self.runner = QASearchRunner(
            index,
            reward_fn,
            top_k=int(cfg.get("top_k", 3)),
            candidate_k=int(cfg.get("candidate_k", 20)),
            candidate_max_per_source=int(cfg.get("candidate_max_per_source", 4)),
            answerability_rerank=bool(cfg.get("answerability_rerank", False)),
            query_expansion=bool(cfg.get("query_expansion", False)),
            max_searches=int(cfg.get("max_searches", 2)),
            max_result_chars=int(cfg.get("max_result_chars", 1500)),
        )
        print(f"文档索引完成：{len(index.chunks)} 个片段，目录 {index.docs_dir}")

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[dict[str, Any]],
    ) -> EnvironmentReturn[dict[str, Any]]:
        results = [
            self.runner.process_turn(log, meta)
            for log, meta in zip(message_log_batch, metadata, strict=False)
        ]
        return EnvironmentReturn(
            observations=[result.observation for result in results],
            metadata=[result.metadata for result in results],
            next_stop_strings=[result.next_stop_strings for result in results],
            rewards=torch.tensor([result.reward for result in results], dtype=torch.float32),
            terminateds=torch.tensor(
                [result.terminated for result in results], dtype=torch.bool
            ),
            answers=[result.answer for result in results],
        )

    def shutdown(self):
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        rewards = batch.get(
            "total_reward", torch.tensor([0.0] * len(batch["idx"]))
        ).float()
        logs = batch.get("message_log", [])
        search_counts = []
        final_flags = []
        question_types = []
        for log in logs:
            assistant_texts = [
                str(message.get("content", ""))
                for message in log
                if message.get("role") == "assistant"
            ]
            search_counts.append(sum("<search>" in text.lower() for text in assistant_texts))
            final_flags.append(any("\\boxed" in text for text in assistant_texts))
            question_types.append(
                qa_type_from_text(
                    "\n".join(str(message.get("content", "")) for message in log)
                )
            )

        reward_values = rewards.detach().cpu().tolist()
        indices = batch["idx"]
        if torch.is_tensor(indices):
            prompt_indices = indices.detach().cpu().tolist()
        else:
            prompt_indices = list(indices)

        metrics = {
            "qa_mean_reward": rewards.mean().item() if len(rewards) else 0.0,
            "qa_perfect_rate": (rewards >= 1.0).float().mean().item()
            if len(rewards)
            else 0.0,
            "qa_format_penalty_rate": (rewards < 0).float().mean().item()
            if len(rewards)
            else 0.0,
            "qa_search_rate": sum(count > 0 for count in search_counts)
            / max(1, len(search_counts)),
            "qa_avg_searches": sum(search_counts) / max(1, len(search_counts)),
            "qa_final_answer_rate": sum(final_flags) / max(1, len(final_flags)),
        }
        metrics.update(
            qa_reward_diagnostics(reward_values, prompt_indices, question_types)
        )
        if self._get_judge_stats is not None:
            judge_stats = self._get_judge_stats(reset=True)
            requests = judge_stats["requested"]
            metrics.update({
                "qa_judge_requests": float(requests),
                "qa_judge_success_rate": (
                    judge_stats["success"] / requests if requests else 0.0
                ),
                "qa_judge_fallback_rate": (
                    judge_stats["fallback"] / requests if requests else 0.0
                ),
            })
        stage = "train" if metrics["qa_multi_sample_group_count"] else "validation"
        print(f"QA诊断[{stage}]：{dict(sorted(metrics.items()))}")
        return batch, metrics
