"""QA 多轮检索的纯 Python 核心。"""

from __future__ import annotations

import heapq
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from common.rewards.qa_reward import FORMAT_PENALTY, extract_boxed

STOP_STRINGS = ["</search>", "</answer>"]

_ASCII_TOKEN = re.compile(r"[a-z0-9][a-z0-9_.+-]*")
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$")
_SEARCH = re.compile(r"<search>\s*(.*?)\s*</search>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class SearchChunk:
    source: str
    text: str


@dataclass(frozen=True)
class SearchHit:
    source: str
    text: str
    score: float


@dataclass(frozen=True)
class TurnResult:
    observation: dict[str, str]
    reward: float
    terminated: bool
    next_stop_strings: list[str] | None
    metadata: dict[str, Any] | None
    answer: str | None


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text)).lower()


def _terms(text: str) -> list[str]:
    """英文按词、中文按二元组切分，兼顾缩写和中文术语。"""
    normalized = _normalize(text)
    terms = _ASCII_TOKEN.findall(normalized)
    for run in _CJK_RUN.findall(normalized):
        if len(run) == 1:
            terms.append(run)
        else:
            terms.extend(run[i : i + 2] for i in range(len(run) - 1))
            if len(run) <= 12:
                terms.append(run)
    return terms


def _split_long(text: str, limit: int, overlap: int = 80) -> Iterable[str]:
    if len(text) <= limit:
        yield text
        return
    step = max(1, limit - min(overlap, limit // 4))
    for start in range(0, len(text), step):
        piece = text[start : start + limit].strip()
        if piece:
            yield piece
        if start + limit >= len(text):
            break


def _markdown_blocks(text: str, fallback_heading: str) -> list[str]:
    heading = fallback_heading
    paragraph: list[str] = []
    blocks: list[str] = []

    def flush() -> None:
        if paragraph:
            body = "\n".join(paragraph).strip()
            if body:
                blocks.append(f"[{heading}]\n{body}")
            paragraph.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = _HEADING.match(line)
        if match:
            flush()
            heading = match.group(1).strip() or fallback_heading
        elif not line:
            flush()
        else:
            paragraph.append(line)
    flush()
    return blocks


class LocalMarkdownIndex:
    """启动时构建一次的确定性 BM25 风格索引。"""

    def __init__(
        self,
        docs_dir: str | Path,
        *,
        chunk_chars: int = 480,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.docs_dir = Path(docs_dir)
        self.chunk_chars = max(160, int(chunk_chars))
        self.k1 = float(k1)
        self.b = float(b)
        self.chunks = self._load_chunks()
        if not self.chunks:
            raise ValueError(f"未在 {self.docs_dir} 找到可索引的 Markdown 内容")
        self._build()

    def _load_chunks(self) -> list[SearchChunk]:
        if not self.docs_dir.is_dir():
            raise FileNotFoundError(f"文档目录不存在: {self.docs_dir}")

        chunks: list[SearchChunk] = []
        for path in sorted(self.docs_dir.rglob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            source = path.relative_to(self.docs_dir).as_posix()
            blocks = _markdown_blocks(text, path.stem)
            pending = ""
            for block in blocks:
                if len(block) > self.chunk_chars:
                    if pending:
                        chunks.append(SearchChunk(source, pending))
                        pending = ""
                    chunks.extend(
                        SearchChunk(source, piece)
                        for piece in _split_long(block, self.chunk_chars)
                    )
                elif not pending:
                    pending = block
                elif len(pending) + 2 + len(block) <= self.chunk_chars:
                    pending += "\n\n" + block
                else:
                    chunks.append(SearchChunk(source, pending))
                    pending = block
            if pending:
                chunks.append(SearchChunk(source, pending))
        return chunks

    def _build(self) -> None:
        self.doc_lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        document_frequency: Counter[str] = Counter()

        for doc_id, chunk in enumerate(self.chunks):
            counts = Counter(_terms(chunk.source + "\n" + chunk.text))
            self.doc_lengths.append(sum(counts.values()))
            for term, count in counts.items():
                self.postings[term].append((doc_id, count))
                document_frequency[term] += 1

        self.avg_doc_length = sum(self.doc_lengths) / max(1, len(self.doc_lengths))
        size = len(self.chunks)
        self.idf = {
            term: math.log(1.0 + (size - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        max_per_source: int = 2,
    ) -> list[SearchHit]:
        query_counts = Counter(_terms(query))
        scores: dict[int, float] = defaultdict(float)
        for term, query_tf in query_counts.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id, tf in self.postings[term]:
                length = self.doc_lengths[doc_id]
                denominator = tf + self.k1 * (
                    1.0 - self.b + self.b * length / self.avg_doc_length
                )
                scores[doc_id] += query_tf * idf * tf * (self.k1 + 1.0) / denominator

        # 每个来源只保留有限候选，避免对近百万命中片段做完整排序。
        source_heaps: dict[str, list[tuple[float, int, int]]] = defaultdict(list)
        for doc_id, score in scores.items():
            source = self.chunks[doc_id].source
            item = (score, -doc_id, doc_id)
            heap = source_heaps[source]
            if len(heap) < max_per_source:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)

        candidates = [item for heap in source_heaps.values() for item in heap]
        selected = heapq.nlargest(
            max(1, int(top_k)), candidates, key=lambda item: (item[0], item[1])
        )
        return [
            SearchHit(
                self.chunks[doc_id].source,
                self.chunks[doc_id].text,
                score,
            )
            for score, _, doc_id in selected
        ]


def _last_assistant_text(message_log: list[dict[str, Any]]) -> str:
    for message in reversed(message_log):
        if message.get("role") == "assistant":
            return str(message.get("content", "")).strip()
    return ""


def _question_text(query: str) -> str:
    match = re.search(r"题目[:：]\s*(.*?)(?:\n\s*\n选项[:：]|\Z)", query, re.DOTALL)
    return match.group(1).strip() if match else query.strip()


def _safe_label(text: str) -> str:
    return str(text).replace("<", "＜").replace(">", "＞")


class QASearchRunner:
    """处理一条轨迹的一轮搜索或最终作答。"""

    def __init__(
        self,
        index: LocalMarkdownIndex,
        reward_fn: Callable[..., list[float]],
        *,
        top_k: int = 3,
        max_searches: int = 2,
        max_result_chars: int = 1500,
    ):
        self.index = index
        self.reward_fn = reward_fn
        self.top_k = max(1, int(top_k))
        self.max_searches = max(1, int(max_searches))
        self.max_result_chars = max(200, int(max_result_chars))

    def _next_action_hint(self, must_answer: bool) -> str:
        if must_answer:
            return (
                "已完成最后一次检索。下一轮不得再检索，必须直接给出最终答案；"
                "使用 answer XML 元素，并用 \\boxed 命令包裹本题真实答案。"
            )
        return (
            "还可检索一次。仅当结果直接包含所问值、数量或关键要点时作答；"
            "若证据不足，请保留题干中的型号、英文缩写、数字或专有名词，"
            "换一组精确词继续检索，禁止凭常识猜测或输出空的 \\boxed{}。"
            "最终答案必须使用 answer XML 元素和 \\boxed 命令，禁止填写占位词。"
        )

    def _format_hits(
        self,
        search_query: str,
        hits: list[SearchHit],
        *,
        must_answer: bool,
    ) -> str:
        if not hits:
            return (
                f"<search_results query=\"{_safe_label(search_query)}\">\n"
                "未找到匹配内容。\n</search_results>\n"
                f"{self._next_action_hint(must_answer)}"
            )

        parts = [f"<search_results query=\"{_safe_label(search_query)}\">"]
        used = 0
        for rank, hit in enumerate(hits, start=1):
            prefix = f"\n[{rank}] 来源: {_safe_label(hit.source)}\n"
            remaining = self.max_result_chars - used
            if remaining <= 0:
                break
            body = hit.text[:remaining]
            parts.append(prefix + body)
            used += len(body)
        parts.append("\n</search_results>\n" + self._next_action_hint(must_answer))
        return "".join(parts)

    def _request_final_answer(
        self,
        metadata: dict[str, Any],
        expected: str,
    ) -> TurnResult:
        if bool(metadata.get("correction_used", False)):
            return TurnResult(
                observation={
                    "role": "environment",
                    "content": "最终答案格式仍不合格，轨迹结束。",
                },
                reward=FORMAT_PENALTY,
                terminated=True,
                next_stop_strings=None,
                metadata=None,
                answer=expected,
            )

        next_metadata = dict(metadata)
        next_metadata["must_answer"] = True
        next_metadata["correction_used"] = True
        return TurnResult(
            observation={
                "role": "environment",
                "content": (
                    "你还有最后一次格式纠正机会。不得继续检索或复述规则；"
                    "只输出一个 answer XML 元素，并用 \\boxed 命令包裹本题真实答案。"
                ),
            },
            reward=0.0,
            terminated=False,
            next_stop_strings=list(STOP_STRINGS),
            metadata=next_metadata,
            answer=None,
        )

    def process_turn(
        self,
        message_log: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> TurnResult:
        assistant_text = _last_assistant_text(message_log)
        expected = str(metadata.get("expected_answer", ""))
        original_query = str(metadata.get("query", ""))

        if extract_boxed(assistant_text) is not None:
            reward = float(
                self.reward_fn([original_query], [assistant_text], [expected])[0]
            )
            return TurnResult(
                observation={"role": "environment", "content": f"最终得分: {reward:.3f}"},
                reward=reward,
                terminated=True,
                next_stop_strings=None,
                metadata=None,
                answer=expected,
            )

        match = _SEARCH.search(assistant_text)
        if match:
            searches = int(metadata.get("searches", 0))
            if bool(metadata.get("must_answer", False)) or searches >= self.max_searches:
                return self._request_final_answer(metadata, expected)

            search_query = match.group(1).strip()
            if not search_query:
                search_query = _question_text(original_query)
            retrieval_query = search_query + "\n" + _question_text(original_query)
            hits = self.index.search(retrieval_query, top_k=self.top_k)
            next_metadata = dict(metadata)
            next_metadata["searches"] = searches + 1
            next_metadata["must_answer"] = searches + 1 >= self.max_searches
            return TurnResult(
                observation={
                    "role": "environment",
                    "content": self._format_hits(
                        search_query,
                        hits,
                        must_answer=next_metadata["must_answer"],
                    ),
                },
                reward=0.0,
                terminated=False,
                next_stop_strings=list(STOP_STRINGS),
                metadata=next_metadata,
                answer=None,
            )

        return self._request_final_answer(metadata, expected)
