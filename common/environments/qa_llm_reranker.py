"""用平台已有指令模型做小规模候选证据语义重排。"""

from __future__ import annotations

import json
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


def parse_query_list(text: str, *, limit: int = 3) -> list[str]:
    """解析 JSON 数组或逐行检索式，去标签、去重并限长。"""
    raw_text = str(text).strip()
    values: list[str] = []
    array_match = re.search(r"\[[\s\S]*?]", raw_text)
    if array_match:
        try:
            parsed = json.loads(array_match.group(0))
            if isinstance(parsed, list):
                values.extend(str(value) for value in parsed)
        except json.JSONDecodeError:
            pass
    if not values:
        values.extend(
            re.sub(r"^\s*(?:[-*]|\d+[.、)])\s*", "", line)
            for line in raw_text.splitlines()
            if line.strip()
        )
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = re.sub(r"</?search>", "", value, flags=re.IGNORECASE).strip()
        cleaned = cleaned.strip("`\"'[] ")[:160]
        key = re.sub(r"\s+", "", cleaned).lower()
        if len(key) >= 2 and key not in seen:
            seen.add(key)
            unique.append(cleaned)
        if len(unique) >= max(1, int(limit)):
            break
    return unique


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
        decoded = self._generate(self._prompt(question, candidates), max_new_tokens=24)
        return (
            parse_rank_ids(
                decoded,
                candidate_count=len(candidates),
                limit=limit,
            ),
            decoded,
        )

    def _generate(self, prompt: str, *, max_new_tokens: int) -> str:
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
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
                max_new_tokens=max(1, int(max_new_tokens)),
                do_sample=False,
            )
        prompt_length = int(inputs["input_ids"].shape[-1])
        return self.processor.decode(
            output[0][prompt_length:],
            skip_special_tokens=True,
        ).strip()

    def rewrite_queries(self, question: str, *, limit: int = 3) -> tuple[list[str], str]:
        prompt = (
            "你是技术文档检索式生成器。根据问题生成三个彼此互补、可直接用于检索资料的简短查询。\n"
            "保留问题中的英文缩写、数字、型号和专有名词；可补充通用同义表达，但不得回答问题或编造具体答案。\n"
            "只返回 JSON 字符串数组，例如 [\"术语 规范\", \"缩写 定义\", \"操作 要求\"]。\n"
            f"问题：{question}"
        )
        decoded = self._generate(prompt, max_new_tokens=96)
        return parse_query_list(decoded, limit=limit), decoded
