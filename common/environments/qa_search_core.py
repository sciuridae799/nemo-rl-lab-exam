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
_EXPECTED_ANSWER = re.compile(r"\s*\[(\w+)]\s*(.*)", re.DOTALL)
_ASCII_SPLIT = re.compile(r"[^a-z0-9]+")
_BLANK_MARKER = re.compile(r"【\s*\d+\s*】|\[\s*\d+\s*]|_{2,}|＿{2,}|\(\s*\)|（\s*）")
_EXAM_CUE = re.compile(
    r"填空题|选择题|判断题|简答题|每题\s*\d*\s*分|^\s*\d+[.、)]",
    re.IGNORECASE | re.MULTILINE,
)
_ANSWER_CUE = re.compile(
    r"是|为|通过|包括|包含|共有|分别|等于|采用|使用|由.{0,24}组成|"
    r"consists?\s+of|includes?|contains?|there\s+(?:is|are)|\b(?:has|have)\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"[\n\r。！？!?；;]+")
_SOURCE_ROLE = re.compile(
    r"(?:参考|标准)?答案|答题卡|试卷|试题|题库|已审核|水印|"
    r"\b(?:answer|solution|exam|quiz)\b|(?:^|[_-])ex$",
    re.IGNORECASE,
)
_ENGLISH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "each",
    "for",
    "how",
    "is",
    "many",
    "of",
    "or",
    "the",
    "there",
    "to",
    "what",
    "which",
}
_QA_TYPE_MARKERS = {
    "single": "一道单选题",
    "multiple": "一道多选题",
    "bool": "一道判断题",
    "fill": "一道填空题",
    "short": "一道简答题",
}
ANSWERABILITY_FEATURE_NAMES = (
    "bm25",
    "question_coverage",
    "ascii_coverage",
    "answer_sentence",
    "answer_cue",
    "new_numeric_value",
    "cloze_bridge",
    "source_answer_cue",
    "source_exam_copy",
    "question_copy",
    "negative_bridge",
)
_ANSWERABILITY_DEFAULT_WEIGHTS = (
    0.35,
    0.20,
    0.10,
    0.20,
    0.10,
    0.05,
    0.75,
    0.08,
    -0.10,
    -0.80,
    -0.25,
)


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


def _terms(text: str, *, expand_ascii: bool = False) -> list[str]:
    """英文按词、中文按二元组切分，兼顾缩写和中文术语。"""
    normalized = _normalize(text)
    terms: list[str] = []
    for token in _ASCII_TOKEN.findall(normalized):
        terms.append(token)
        if expand_ascii:
            terms.extend(
                piece
                for piece in _ASCII_SPLIT.split(token)
                if len(piece) >= 2 or any(char.isdigit() for char in piece)
            )
    for run in _CJK_RUN.findall(normalized):
        if len(run) == 1:
            terms.append(run)
        else:
            terms.extend(run[i : i + 2] for i in range(len(run) - 1))
            if len(run) <= 12:
                terms.append(run)
    return terms


def compact_text(text: str) -> str:
    """NFKC、小写并只保留字母数字和汉字，用于稳健的证据匹配。"""
    return "".join(char for char in _normalize(text) if char.isalnum())


def source_family_key(source: str) -> str:
    """去掉文档角色词，识别同目录下的试卷/答案等兄弟文件。"""
    path = Path(str(source))
    stem = compact_text(_SOURCE_ROLE.sub("", _normalize(path.stem)))
    parent = compact_text(path.parent.as_posix())
    return f"{parent}/{stem}" if stem else ""


def source_has_answer_role(source: str) -> bool:
    """识别通用的答案/讲义文件角色，不依赖具体部门或设备词。"""
    return bool(
        re.search(
            r"参考答案|标准答案|答案|讲义|教材|manual|solution|answer|training",
            str(source),
            re.IGNORECASE,
        )
    )


def _informative_terms(text: str) -> set[str]:
    terms = set(_terms(text))
    return {term for term in terms if not (term.isascii() and term in _ENGLISH_STOPWORDS)}


def _term_coverage(question: str, text: str) -> float:
    question_terms = _informative_terms(_BLANK_MARKER.sub("", question))
    if not question_terms:
        return 0.0
    return len(question_terms & set(_terms(text))) / len(question_terms)


def _cloze_bridge_signal(question: str, text: str) -> float:
    """判断片段是否在填空左右文之间真正给出了内容，而非复刻空题。"""
    parts = _BLANK_MARKER.split(_normalize(question))
    if len(parts) < 2:
        return 0.0

    compact_passage = compact_text(text)
    signals: list[float] = []
    for left, right in zip(parts, parts[1:], strict=False):
        left_compact = compact_text(left)
        right_compact = compact_text(right)
        if not left_compact or not right_compact:
            continue
        # OCR、换行和中英文混排会让固定长度锚点偶尔失配；从长到短
        # 尝试多个锚点，并遍历少量左侧出现位置，避免被题面复刻遮蔽。
        for anchor_length in (18, 14, 10, 7, 5):
            left_anchor = left_compact[-min(anchor_length, len(left_compact)) :]
            right_anchor = right_compact[: min(anchor_length, len(right_compact))]
            search_from = 0
            while True:
                left_pos = compact_passage.find(left_anchor, search_from)
                if left_pos < 0:
                    break
                gap_start = left_pos + len(left_anchor)
                right_pos = compact_passage.find(right_anchor, gap_start)
                if right_pos >= 0:
                    gap_length = right_pos - gap_start
                    if gap_length == 0:
                        signals.append(-1.0)
                    elif gap_length <= 48:
                        signals.append(1.0)
                    elif gap_length <= 120:
                        signals.append(0.25)
                search_from = left_pos + 1
                # 同一锚点出现过多时不让一个长题面拖慢每轮检索。
                if search_from > 4096:
                    break
            if any(signal > 0 for signal in signals):
                break

    if any(signal > 0 for signal in signals):
        return max(signals)
    return min(signals, default=0.0)


def question_copy_score(question: str, text: str) -> float:
    """识别只复刻题面、没有补出答案的试卷型片段。"""
    cleaned_question = _BLANK_MARKER.sub("", question)
    question_compact = compact_text(cleaned_question)
    if len(question_compact) < 6:
        return 0.0

    coverage = _term_coverage(question, text)
    exact_copy = question_compact in compact_text(text)
    exam_like = bool(_EXAM_CUE.search(text))
    bridge = _cloze_bridge_signal(question, text)
    score = 0.0
    if exam_like and coverage >= 0.65:
        score = max(score, coverage)
    if exact_copy and exam_like:
        score = 1.0
    if bridge < 0:
        score = 1.0
    return score


def evidence_progress_coverage(
    question: str,
    expected: str,
    hits: list[SearchHit],
) -> float:
    """训练时衡量检索新增了多少开放题答案证据，不把 gold 暴露给模型。"""
    match = _EXPECTED_ANSWER.match(str(expected))
    if not match or match.group(1) not in {"fill", "short"}:
        return 0.0

    question_compact = compact_text(question)
    items: list[list[str]] = []
    for raw_item in match.group(2).split("|||"):
        alternatives = [
            compact_text(part)
            for part in re.split(r"[/／]", raw_item)
            if compact_text(part)
        ]
        # 题干本身已经出现的词不算检索进展，避免奖励原题/空白试卷复刻。
        novel = [
            alternative
            for alternative in alternatives
            if alternative not in question_compact
        ]
        if novel:
            items.append(novel)

    if not items or not hits:
        return 0.0

    passage = "\n".join(hit.source + "\n" + hit.text for hit in hits)
    compact_passage = compact_text(passage)
    ascii_tokens = set(re.findall(r"[a-z0-9]+", _normalize(passage)))
    covered = 0
    for alternatives in items:
        matched = any(
            (len(alternative) >= 2 and alternative in compact_passage)
            or (alternative.isdigit() and alternative in ascii_tokens)
            for alternative in alternatives
        )
        covered += int(matched)
    return covered / len(items)


def _best_answer_sentence_score(question: str, text: str) -> float:
    question_terms = _informative_terms(_BLANK_MARKER.sub("", question))
    if not question_terms:
        return 0.0
    best = 0.0
    for sentence in _SENTENCE_SPLIT.split(text):
        sentence_terms = set(_terms(sentence))
        coverage = len(question_terms & sentence_terms) / len(question_terms)
        cue_bonus = 0.2 if _ANSWER_CUE.search(sentence) else 0.0
        best = max(best, min(1.0, coverage + cue_bonus))
    return best


def _snippet_relevance_score(question: str, text: str) -> float:
    """给候选片段中的短窗口打分，不依赖 gold 或题目特例。"""
    bridge = _cloze_bridge_signal(question, text)
    return (
        0.45 * _term_coverage(question, text)
        + 0.30 * _best_answer_sentence_score(question, text)
        + 0.75 * max(0.0, bridge)
        + 0.15 * float(bool(_ANSWER_CUE.search(text)))
        - 0.80 * question_copy_score(question, text)
        - 0.20 * max(0.0, -bridge)
    )


def _snippet_windows(text: str, limit: int) -> list[str]:
    """生成句子边界与重叠定长窗口，兼顾长表格和普通段落。"""
    stripped = str(text).strip()
    if len(stripped) <= limit:
        return [stripped] if stripped else []

    spans = [
        span.strip()
        for span in re.split(r"(?<=[。！？!?；;])|\n+", stripped)
        if span.strip()
    ]
    windows = [span for span in spans if len(span) <= limit]
    step = max(1, limit // 2)
    windows.extend(
        stripped[start : start + limit].strip()
        for start in range(0, len(stripped), step)
        if stripped[start : start + limit].strip()
    )
    return windows


def extract_answerable_snippets(
    question: str,
    hits: list[SearchHit],
    *,
    max_chars: int = 140,
) -> list[SearchHit]:
    """把每个候选块压缩成一个问题相关短片段，以相同正文预算容纳更多来源。"""
    limit = max(60, int(max_chars))
    snippets: list[SearchHit] = []
    for hit in hits:
        windows = _snippet_windows(hit.text, limit)
        if not windows:
            continue
        best_position = max(
            range(len(windows)),
            key=lambda position: (
                _snippet_relevance_score(question, windows[position]),
                -position,
            ),
        )
        snippets.append(SearchHit(hit.source, windows[best_position], hit.score))
    return snippets


def _content_similarity(left: str, right: str) -> float:
    left_terms = set(_terms(left))
    right_terms = set(_terms(right))
    union = left_terms | right_terms
    return len(left_terms & right_terms) / len(union) if union else 0.0


def answerability_feature_rows(
    question: str,
    hits: list[SearchHit],
) -> list[tuple[float, ...]]:
    """提取 gold 无关的候选特征，供固定规则与训练集监督重排共用。"""
    if not hits:
        return []

    raw_scores = [hit.score for hit in hits]
    low, high = min(raw_scores), max(raw_scores)
    question_ascii = {
        term
        for term in _ASCII_TOKEN.findall(_normalize(question))
        if term not in _ENGLISH_STOPWORDS
        and (len(term) >= 3 or any(char.isdigit() for char in term))
    }
    question_numbers = {
        term
        for term in _ASCII_TOKEN.findall(_normalize(question))
        if any(char.isdigit() for char in term)
    }

    rows: list[tuple[float, ...]] = []
    for hit in hits:
        bm25 = (hit.score - low) / (high - low) if high > low else 1.0
        passage = hit.source + "\n" + hit.text
        ascii_terms = set(_ASCII_TOKEN.findall(_normalize(passage)))
        ascii_coverage = (
            len(question_ascii & ascii_terms) / len(question_ascii)
            if question_ascii
            else 0.0
        )
        new_values = {
            term
            for term in ascii_terms - question_numbers
            if any(char.isdigit() for char in term)
        }
        bridge = _cloze_bridge_signal(question, hit.text)
        copy_score = question_copy_score(question, hit.text)
        source_answer_cue = bool(
            re.search(
                r"参考答案|答案|answer|solution|manual|training",
                hit.source,
                re.IGNORECASE,
            )
        )
        source_exam_cue = bool(
            re.search(r"试卷|试题|题库|exam|quiz", hit.source, re.IGNORECASE)
        )
        rows.append(
            (
                bm25,
                _term_coverage(question, passage),
                ascii_coverage,
                _best_answer_sentence_score(question, hit.text),
                float(bool(_ANSWER_CUE.search(hit.text))),
                min(1.0, len(new_values) / 2.0),
                max(0.0, bridge),
                float(source_answer_cue),
                float(source_exam_cue and copy_score >= 0.5),
                copy_score,
                max(0.0, -bridge),
            )
        )
    return rows


def rerank_answerable_hits(
    question: str,
    hits: list[SearchHit],
    *,
    top_k: int,
    baseline_hits: list[SearchHit] | None = None,
    weights: tuple[float, ...] | None = None,
) -> list[SearchHit]:
    """在少量 BM25 候选中优先选择可回答、非重复的证据片段。"""
    if not hits:
        return []

    active_weights = tuple(weights or _ANSWERABILITY_DEFAULT_WEIGHTS)
    if len(active_weights) != len(ANSWERABILITY_FEATURE_NAMES):
        raise ValueError("answerability weights 长度与特征不一致")
    scored: list[tuple[float, int, SearchHit]] = []
    feature_rows = answerability_feature_rows(question, hits)
    for position, (hit, features) in enumerate(zip(hits, feature_rows, strict=True)):
        score = sum(
            weight * value
            for weight, value in zip(
                active_weights,
                features,
                strict=True,
            )
        )
        scored.append((score, position, SearchHit(hit.source, hit.text, score)))

    selected: list[SearchHit] = []
    remaining = list(scored)
    while remaining and len(selected) < max(1, int(top_k)):
        best_index = max(
            range(len(remaining)),
            key=lambda idx: (
                remaining[idx][0]
                - 0.30
                * max(
                    (_content_similarity(remaining[idx][2].text, chosen.text) for chosen in selected),
                    default=0.0,
                ),
                -remaining[idx][1],
            ),
        )
        _, _, best_hit = remaining.pop(best_index)
        selected.append(best_hit)

    if baseline_hits and selected:
        baseline = baseline_hits[0]
        baseline_key = (baseline.source, baseline.text)
        selected_keys = {(hit.source, hit.text) for hit in selected}
        best_candidate = selected[0]
        # 没有明确填空左右文桥接证据时，保留原始 BM25 第一候选作为负回退；
        # 这样一个启发式重排不会把已有的可用证据替换成泛泛而谈的段落。
        can_promote = _cloze_bridge_signal(question, best_candidate.text) > 0.5
        if baseline_key not in selected_keys and not can_promote:
            selected = [baseline] + [
                hit
                for hit in selected
                if (hit.source, hit.text) != baseline_key
            ]
            selected = selected[: max(1, int(top_k))]
    return selected


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
        expand_ascii_tokens: bool = False,
    ):
        self.docs_dir = Path(docs_dir)
        self.chunk_chars = max(160, int(chunk_chars))
        self.k1 = float(k1)
        self.b = float(b)
        self.expand_ascii_tokens = bool(expand_ascii_tokens)
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
        self.source_doc_ids: dict[str, list[int]] = defaultdict(list)
        self.doc_source_positions: list[int] = []
        self.chunk_doc_ids: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.family_sources: dict[str, set[str]] = defaultdict(set)
        document_frequency: Counter[str] = Counter()

        for doc_id, chunk in enumerate(self.chunks):
            self.doc_source_positions.append(len(self.source_doc_ids[chunk.source]))
            self.source_doc_ids[chunk.source].append(doc_id)
            self.chunk_doc_ids[(chunk.source, chunk.text)].append(doc_id)
            family = source_family_key(chunk.source)
            if family:
                self.family_sources[family].add(chunk.source)
            counts = Counter(
                _terms(
                    chunk.source + "\n" + chunk.text,
                    expand_ascii=self.expand_ascii_tokens,
                )
            )
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
        prefix_terms: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for term, idf in self.idf.items():
            if len(term) >= 4 and re.fullmatch(r"[a-z0-9]+", term):
                prefix_terms[term[:4]].append((idf, term))
        self.prefix_terms = {
            prefix: [term for _, term in sorted(values, reverse=True)[:8]]
            for prefix, values in prefix_terms.items()
        }

    def _score_query(self, query: str) -> dict[int, float]:
        """计算全索引 BM25 分数，供普通搜索和结构扩展复用。"""
        query_counts = Counter(
            _terms(query, expand_ascii=self.expand_ascii_tokens)
        )
        scores: dict[int, float] = defaultdict(float)
        weighted_terms: list[tuple[str, float]] = []
        for term, query_tf in query_counts.items():
            weighted_terms.append((term, float(query_tf)))
            if self.expand_ascii_tokens and len(term) >= 4 and re.fullmatch(
                r"[a-z0-9]+", term
            ):
                weighted_terms.extend(
                    (alias, float(query_tf) * 0.2)
                    for alias in self.prefix_terms.get(term[:4], [])
                    if alias != term
                )
        for term, query_tf in weighted_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id, tf in self.postings[term]:
                length = self.doc_lengths[doc_id]
                denominator = tf + self.k1 * (
                    1.0 - self.b + self.b * length / self.avg_doc_length
                )
                scores[doc_id] += (
                    query_tf * idf * tf * (self.k1 + 1.0) / denominator
                )
        return scores

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        max_per_source: int = 2,
        candidate_k: int | None = None,
        rerank_question: str | None = None,
        preserve_top_hit: bool = False,
    ) -> list[SearchHit]:
        scores = self._score_query(query)

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
        candidate_limit = max(1, int(candidate_k or top_k), int(top_k))
        selected = heapq.nlargest(
            candidate_limit, candidates, key=lambda item: (item[0], item[1])
        )
        hits = [
            SearchHit(
                self.chunks[doc_id].source,
                self.chunks[doc_id].text,
                score,
            )
            for score, _, doc_id in selected
        ]
        if rerank_question:
            return rerank_answerable_hits(
                rerank_question,
                hits,
                top_k=top_k,
                baseline_hits=hits[:1] if preserve_top_hit else None,
            )
        return hits[: max(1, int(top_k))]

    def search_union(
        self,
        queries: list[str],
        *,
        candidate_k: int = 20,
        max_per_source: int = 2,
    ) -> list[SearchHit]:
        """合并多个题干视角的 BM25 候选，供后续可回答性重排。"""
        merged: dict[tuple[str, str], SearchHit] = {}
        for query in queries:
            if not str(query).strip():
                continue
            for hit in self.search(
                query,
                top_k=candidate_k,
                max_per_source=max_per_source,
            ):
                key = (hit.source, hit.text)
                previous = merged.get(key)
                if previous is None or hit.score > previous.score:
                    merged[key] = hit
        return sorted(
            merged.values(),
            key=lambda hit: hit.score,
            reverse=True,
        )[: max(1, int(candidate_k)) * max(1, len(queries))]

    def expand_structural_candidates(
        self,
        query: str,
        hits: list[SearchHit],
        *,
        neighbor_window: int = 1,
        sibling_k: int = 4,
        max_seed_hits: int = 160,
        include_aligned_siblings: bool = False,
    ) -> list[SearchHit]:
        """补充同源相邻块和同文件族兄弟文档，不改变模型查询。

        ``include_aligned_siblings`` 会将试卷片段的序号对齐到同族答案/讲义文档，
        即使对齐片段本身没有题干词也保留为候选。默认关闭以保持旧行为。"""
        merged = {(hit.source, hit.text): hit for hit in hits}
        query_scores = self._score_query(query)
        sibling_sources: set[str] = set()
        aligned_positions: dict[str, list[tuple[int, float]]] = defaultdict(list)

        for hit in hits[: max(1, int(max_seed_hits))]:
            doc_ids = self.chunk_doc_ids.get((hit.source, hit.text), [])
            if doc_ids:
                doc_id = doc_ids[0]
                source_ids = self.source_doc_ids[hit.source]
                position = self.doc_source_positions[doc_id]
                family = source_family_key(hit.source)
                if family:
                    aligned_positions[family].append((position, hit.score))
                for offset in range(-max(0, int(neighbor_window)), max(0, int(neighbor_window)) + 1):
                    neighbor_position = position + offset
                    if offset == 0 or not 0 <= neighbor_position < len(source_ids):
                        continue
                    neighbor_id = source_ids[neighbor_position]
                    chunk = self.chunks[neighbor_id]
                    key = (chunk.source, chunk.text)
                    score = max(query_scores.get(neighbor_id, 0.0), hit.score * 0.85)
                    previous = merged.get(key)
                    if previous is None or score > previous.score:
                        merged[key] = SearchHit(chunk.source, chunk.text, score)

            family = source_family_key(hit.source)
            if family:
                sibling_sources.update(self.family_sources.get(family, set()))

        sibling_sources.difference_update(hit.source for hit in hits)
        for source in sibling_sources:
            family = source_family_key(source)
            aligned = aligned_positions.get(family, [])
            aligned_candidates: list[tuple[float, int, int]] = []
            if include_aligned_siblings and aligned:
                source_ids = self.source_doc_ids[source]
                for position, seed_score in aligned:
                    for offset in range(
                        -max(0, int(neighbor_window)),
                        max(0, int(neighbor_window)) + 1,
                    ):
                        sibling_position = position + offset
                        if not 0 <= sibling_position < len(source_ids):
                            continue
                        doc_id = source_ids[sibling_position]
                        query_score = query_scores.get(doc_id, 0.0)
                        aligned_score = max(query_score, seed_score * 0.72)
                        aligned_candidates.append(
                            (
                                float(source_has_answer_role(source)),
                                aligned_score,
                                doc_id,
                            )
                        )
            if aligned_candidates:
                ranked_ids = [
                    doc_id
                    for _, _, doc_id in sorted(
                        aligned_candidates,
                        key=lambda item: (item[0], item[1], -item[2]),
                        reverse=True,
                    )[: max(1, int(sibling_k))]
                ]
            else:
                ranked_ids = heapq.nlargest(
                    max(1, int(sibling_k)),
                    self.source_doc_ids[source],
                    key=lambda doc_id: (query_scores.get(doc_id, 0.0), -doc_id),
                )
            for doc_id in ranked_ids:
                score = query_scores.get(doc_id, 0.0)
                if include_aligned_siblings and aligned:
                    seed_scores = [
                        seed_score
                        for position, seed_score in aligned
                        if abs(self.doc_source_positions[doc_id] - position)
                        <= max(0, int(neighbor_window))
                    ]
                    score = max(score, max(seed_scores, default=0.0) * 0.72)
                if score <= 0.0:
                    continue
                chunk = self.chunks[doc_id]
                key = (chunk.source, chunk.text)
                previous = merged.get(key)
                if previous is None or score > previous.score:
                    merged[key] = SearchHit(chunk.source, chunk.text, score)

        return sorted(merged.values(), key=lambda hit: hit.score, reverse=True)


def _last_assistant_text(message_log: list[dict[str, Any]]) -> str:
    for message in reversed(message_log):
        if message.get("role") == "assistant":
            return str(message.get("content", "")).strip()
    return ""


def _question_text(query: str) -> str:
    match = re.search(r"题目[:：]\s*(.*?)(?:\n\s*\n选项[:：]|\Z)", query, re.DOTALL)
    return match.group(1).strip() if match else query.strip()


def build_query_variants(question: str, search_query: str = "") -> list[str]:
    """从题干构造少量通用检索视角，不引入答案或题目特例。"""
    question = str(question).strip()
    variants: list[str] = []
    if search_query.strip():
        variants.append(search_query.strip() + "\n" + question)
    variants.append(question)

    ascii_tokens: list[str] = []
    for token in _ASCII_TOKEN.findall(_normalize(question)):
        pieces = [piece for piece in _ASCII_SPLIT.split(token) if piece]
        ascii_tokens.extend(pieces or [token])
    ascii_tokens = list(
        dict.fromkeys(token for token in ascii_tokens if len(token) >= 2)
    )
    if ascii_tokens:
        variants.append(" ".join(ascii_tokens))

    blank_parts = [part.strip() for part in _BLANK_MARKER.split(question)]
    variants.extend(part for part in blank_parts if len(compact_text(part)) >= 4)

    for run in _CJK_RUN.findall(question):
        if len(run) <= 12:
            variants.append(run)
        else:
            variants.extend(run[start : start + 8] for start in range(0, len(run), 6))

    unique: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        key = compact_text(variant)
        if len(key) >= 3 and key not in seen:
            seen.add(key)
            unique.append(variant)
    return unique[:8]


def _safe_label(text: str) -> str:
    return str(text).replace("<", "＜").replace(">", "＞")


def qa_type_from_text(text: str) -> str:
    """从项目固定题面前缀识别题型。"""
    for name, marker in _QA_TYPE_MARKERS.items():
        if marker in str(text):
            return name
    return "unknown"


def filter_qa_rows_by_type(
    rows: Iterable[dict[str, Any]],
    allowed_question_types: Iterable[str] | None,
    *,
    input_key: str = "query",
) -> list[dict[str, Any]]:
    """按显式题型白名单筛选训练行；空白名单保持原顺序和内容。"""
    requested = {str(name).strip() for name in (allowed_question_types or ()) if str(name).strip()}
    if not requested:
        return list(rows)
    invalid = requested - set(_QA_TYPE_MARKERS)
    if invalid:
        raise ValueError(f"未知 QA 题型：{sorted(invalid)}")
    filtered = [
        row
        for row in rows
        if qa_type_from_text(str(row.get(input_key, ""))) in requested
    ]
    if not filtered:
        raise ValueError(f"题型过滤后训练集为空：允许 {sorted(requested)}")
    return filtered


def qa_loss_multiplier(
    question: str,
    *,
    is_training: bool,
    open_loss_multiplier: float = 1.0,
) -> float:
    """可选地降低训练开放题的更新权重，验证和封闭题保持完整权重。"""
    if not is_training or qa_type_from_text(question) not in {"fill", "short"}:
        return 1.0
    return max(0.0, min(1.0, float(open_loss_multiplier)))


def qa_reward_diagnostics(
    rewards: list[float],
    prompt_indices: list[int],
    question_types: list[str],
) -> dict[str, float]:
    """统计各题型奖励与多生成 prompt 的零方差比例。"""
    if not (len(rewards) == len(prompt_indices) == len(question_types)):
        raise ValueError("诊断输入长度不一致")

    metrics: dict[str, float] = {}
    for name in _QA_TYPE_MARKERS:
        values = [
            reward
            for reward, question_type in zip(rewards, question_types, strict=False)
            if question_type == name
        ]
        count = len(values)
        metrics[f"qa_type_{name}_count"] = float(count)
        metrics[f"qa_type_{name}_mean_reward"] = (
            sum(values) / count if count else 0.0
        )
        metrics[f"qa_type_{name}_zero_rate"] = (
            sum(value == 0.0 for value in values) / count if count else 0.0
        )

    grouped: dict[int, list[float]] = defaultdict(list)
    for prompt_idx, reward in zip(prompt_indices, rewards, strict=False):
        grouped[int(prompt_idx)].append(float(reward))
    multi_sample_groups = [values for values in grouped.values() if len(values) > 1]
    zero_variance = sum(
        max(values) - min(values) <= 1.0e-8 for values in multi_sample_groups
    )
    group_count = len(multi_sample_groups)
    metrics["qa_multi_sample_group_count"] = float(group_count)
    metrics["qa_zero_variance_group_rate"] = (
        zero_variance / group_count if group_count else 0.0
    )
    metrics["qa_effective_group_rate"] = (
        1.0 - metrics["qa_zero_variance_group_rate"] if group_count else 0.0
    )
    return metrics


class QASearchRunner:
    """处理一条轨迹的一轮搜索或最终作答。"""

    def __init__(
        self,
        index: LocalMarkdownIndex,
        reward_fn: Callable[..., list[float]],
        *,
        top_k: int = 3,
        candidate_k: int = 20,
        candidate_max_per_source: int = 4,
        answerability_rerank: bool = False,
        query_expansion: bool = False,
        structural_expansion: bool = False,
        aligned_sibling_expansion: bool = False,
        max_searches: int = 2,
        max_result_chars: int = 1500,
        evidence_reward_scale: float = 0.0,
        qa_memory: Any | None = None,
        qa_memory_top_k: int = 5,
        qa_memory_min_similarity: float = 0.15,
        qa_memory_max_chars: int = 900,
    ):
        self.index = index
        self.reward_fn = reward_fn
        self.top_k = max(1, int(top_k))
        self.candidate_k = max(self.top_k, int(candidate_k))
        self.candidate_max_per_source = max(1, int(candidate_max_per_source))
        self.answerability_rerank = bool(answerability_rerank)
        self.query_expansion = bool(query_expansion)
        self.structural_expansion = bool(structural_expansion)
        self.aligned_sibling_expansion = bool(aligned_sibling_expansion)
        self.max_searches = max(1, int(max_searches))
        self.max_result_chars = max(200, int(max_result_chars))
        self.evidence_reward_scale = max(0.0, float(evidence_reward_scale))
        self.qa_memory = qa_memory
        self.qa_memory_top_k = max(1, int(qa_memory_top_k))
        self.qa_memory_min_similarity = max(0.0, float(qa_memory_min_similarity))
        self.qa_memory_max_chars = max(200, int(qa_memory_max_chars))

    def _next_action_hint(self, must_answer: bool) -> str:
        if must_answer:
            return (
                "已完成最后一次检索。下一轮不得再检索，必须直接给出最终答案；"
                "使用 answer XML 元素，并用 \\boxed 命令包裹本题真实答案。"
            )
        return (
            "还可检索一次；若已有依据，请直接给出最终答案。"
            "最终答案必须使用 answer XML 元素和 \\boxed 命令，禁止填写占位词。"
        )

    def _format_hits(
        self,
        search_query: str,
        hits: list[SearchHit],
        *,
        must_answer: bool,
        memory_hits: list[Any] | None = None,
    ) -> str:
        if not hits and not memory_hits:
            return (
                f"<search_results query=\"{_safe_label(search_query)}\">\n"
                "未找到匹配内容。\n</search_results>\n"
                f"{self._next_action_hint(must_answer)}"
            )

        parts = [f"<search_results query=\"{_safe_label(search_query)}\">"]
        if memory_hits:
            parts.append(
                "\n<memory_examples>\n"
                "以下仅为相似训练题参考；当前题选项可能不同，必须按当前题语义重新映射，不能机械复制字母。"
            )
            memory_used = 0
            for rank, memory_hit in enumerate(memory_hits, start=1):
                entry = (
                    f"\n[相似题{rank} 相似度={float(memory_hit.similarity):.3f}]\n"
                    f"题目：{memory_hit.question}\n训练集标准答案：{memory_hit.answer}\n"
                )
                remaining = self.qa_memory_max_chars - memory_used
                if remaining <= 0:
                    break
                parts.append(entry[:remaining])
                memory_used += min(len(entry), remaining)
            parts.append("</memory_examples>")
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
            if (
                reward >= 0.0
                and self.evidence_reward_scale > 0.0
                and bool(metadata.get("is_training", False))
                and qa_type_from_text(original_query) in {"fill", "short"}
            ):
                reward = min(
                    1.0,
                    reward
                    + self.evidence_reward_scale
                    * float(metadata.get("evidence_coverage", 0.0)),
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
            question = _question_text(original_query)
            retrieval_query = search_query + "\n" + question
            rerank_question = (
                question
                if self.answerability_rerank and qa_type_from_text(original_query) in {"fill", "short"}
                else None
            )
            baseline_hits = self.index.search(retrieval_query, top_k=self.top_k)
            if rerank_question:
                queries = (
                    build_query_variants(question, search_query)
                    if self.query_expansion
                    else [retrieval_query]
                )
                candidate_hits = self.index.search_union(
                    queries,
                    candidate_k=self.candidate_k,
                    max_per_source=self.candidate_max_per_source,
                )
                if self.structural_expansion:
                    candidate_hits = self.index.expand_structural_candidates(
                        question,
                        candidate_hits,
                        include_aligned_siblings=self.aligned_sibling_expansion,
                    )
                hits = rerank_answerable_hits(
                    question,
                    candidate_hits,
                    top_k=self.top_k,
                    baseline_hits=baseline_hits,
                )
            else:
                hits = baseline_hits
            memory_hits = (
                self.qa_memory.search(
                    original_query,
                    top_k=self.qa_memory_top_k,
                    min_similarity=self.qa_memory_min_similarity,
                )
                if self.qa_memory is not None
                else []
            )
            next_metadata = dict(metadata)
            next_metadata["searches"] = searches + 1
            next_metadata["must_answer"] = searches + 1 >= self.max_searches
            if (
                self.evidence_reward_scale > 0.0
                and bool(metadata.get("is_training", False))
                and qa_type_from_text(original_query) in {"fill", "short"}
            ):
                previous_coverage = float(metadata.get("evidence_coverage", 0.0))
                current_coverage = max(
                    previous_coverage,
                    evidence_progress_coverage(question, expected, hits),
                )
                next_metadata["evidence_coverage"] = current_coverage
                next_metadata["evidence_reward_total"] = (
                    self.evidence_reward_scale * current_coverage
                )
            return TurnResult(
                observation={
                    "role": "environment",
                    "content": self._format_hits(
                        search_query,
                        hits,
                        must_answer=next_metadata["must_answer"],
                        memory_hits=memory_hits,
                    ),
                },
                reward=0.0,
                terminated=False,
                next_stop_strings=list(STOP_STRINGS),
                metadata=next_metadata,
                answer=None,
            )

        return self._request_final_answer(metadata, expected)
