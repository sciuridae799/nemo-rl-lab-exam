"""基于训练题的非参数 QA 记忆检索；只向模型暴露训练数据。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.environments.qa_search_core import (
    _question_text,
    compact_text,
    qa_type_from_text,
)


@dataclass(frozen=True)
class QAMemoryHit:
    question: str
    answer: str
    similarity: float


def _display_question(query: str) -> str:
    match = re.search(r"题目[:：]\s*(.*)", str(query), flags=re.DOTALL)
    return (match.group(1) if match else str(query)).strip()


def _answer_payload(expected: str) -> str:
    match = re.match(r"\s*\[\w+]\s*(.*)", str(expected), flags=re.DOTALL)
    return (match.group(1) if match else str(expected)).strip().replace("|||", "; ")


class QAMemoryIndex:
    """按题型建立字符 TF-IDF 索引，检索相似训练题及其答案。"""

    def __init__(
        self,
        path: str | Path,
        *,
        input_key: str = "query",
        output_key: str = "expected_answer",
        allowed_types: tuple[str, ...] = ("multiple", "bool"),
    ):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:  # pragma: no cover - 集群依赖
            raise RuntimeError("QA memory 需要 scikit-learn") from exc

        rows: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    rows.append(json.loads(line))

        self._indices: dict[str, tuple[Any, Any, list[QAMemoryHit]]] = {}
        allowed = {str(value) for value in allowed_types}
        for question_type in sorted(allowed):
            unique: dict[tuple[str, str], tuple[str, QAMemoryHit]] = {}
            for row in rows:
                query = str(row.get(input_key, ""))
                if qa_type_from_text(query) != question_type:
                    continue
                expected = str(row.get(output_key, ""))
                question = _question_text(query)
                answer = _answer_payload(expected)
                key = (compact_text(question), compact_text(answer))
                if key[0] and key[1] and key not in unique:
                    unique[key] = (
                        compact_text(question),
                        QAMemoryHit(
                            question=_display_question(query),
                            answer=answer,
                            similarity=0.0,
                        ),
                    )
            entries = list(unique.values())
            questions = [entry[0] for entry in entries]
            hits = [entry[1] for entry in entries]
            if not hits:
                continue
            vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 5),
                min_df=1,
                max_features=80_000,
                sublinear_tf=True,
            )
            matrix = vectorizer.fit_transform(questions)
            self._indices[question_type] = (vectorizer, matrix, hits)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        min_similarity: float = 0.15,
    ) -> list[QAMemoryHit]:
        question_type = qa_type_from_text(query)
        indexed = self._indices.get(question_type)
        if indexed is None:
            return []
        vectorizer, matrix, hits = indexed
        query_vector = vectorizer.transform([compact_text(_question_text(query))])
        similarities = (matrix @ query_vector.T).toarray().ravel()
        positions = sorted(
            range(len(similarities)),
            key=lambda position: (float(similarities[position]), -position),
            reverse=True,
        )
        selected: list[QAMemoryHit] = []
        for position in positions:
            similarity = float(similarities[position])
            if similarity < float(min_similarity):
                break
            hit = hits[position]
            selected.append(
                QAMemoryHit(hit.question, hit.answer, similarity)
            )
            if len(selected) >= max(1, int(top_k)):
                break
        return selected
