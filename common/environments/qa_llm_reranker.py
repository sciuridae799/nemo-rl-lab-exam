"""用平台已有指令模型做小规模候选证据语义重排。"""

from __future__ import annotations

import re
from typing import Any


def parse_rank_ids(text: str, *, candidate_count: int, limit: int = 3) -> list[int]:
    """从模型短输出中提取 1-based 编号，并返回去重后的 0-based 下标。"""
    selected: list[int] = []
    for raw in re.findall(r"\d+", str(text)):
        position = int(raw) - 1
        if 0 <= position < int(candidate_count) and position not in selected:
            selected.append(position)
        if len(selected) >= max(1, int(limit)):
            break
    return selected


class QwenCandidateReranker:
    """文本候选批内重排器；模型仅看到问题和候选，不接触期望答案。"""

    def __init__(self, model_path: str):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model.eval()

    @staticmethod
    def _prompt(question: str, candidates: list[tuple[str, str]]) -> str:
        lines = [
            "你是检索证据重排器。判断哪些候选直接包含回答问题所需的事实。",
            "不要回答问题，不要补充常识；只返回最多三个候选编号，例如 [2,5,9]。",
            "若没有候选包含答案依据，返回 []。",
            f"问题：{question}",
            "候选：",
        ]
        for index, (source, text) in enumerate(candidates, start=1):
            lines.append(f"[{index}] 来源：{source}\n{text}")
        return "\n".join(lines)

    def select(
        self,
        question: str,
        candidates: list[tuple[str, str]],
        *,
        limit: int = 3,
    ) -> tuple[list[int], str]:
        if not candidates:
            return [], "[]"
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._prompt(question, candidates)}
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=24,
                do_sample=False,
            )
        prompt_length = int(inputs["input_ids"].shape[-1])
        decoded = self.processor.decode(
            output[0][prompt_length:],
            skip_special_tokens=True,
        ).strip()
        return (
            parse_rank_ids(
                decoded,
                candidate_count=len(candidates),
                limit=limit,
            ),
            decoded,
        )
