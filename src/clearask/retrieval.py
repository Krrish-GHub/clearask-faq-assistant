from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Match:
    faq_id: str
    answer: str
    score: float


class FAQRetriever:
    """Small, inspectable TF-IDF retriever; it does not generate new answers."""

    def __init__(self, faqs: list[dict[str, Any]], analyzer: str = "word", threshold: float = 0.22):
        if analyzer not in {"word", "char"}:
            raise ValueError("analyzer must be 'word' or 'char'")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if not faqs:
            raise ValueError("at least one FAQ is required")
        self.faqs = faqs
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=(1, 2) if analyzer == "word" else (3, 5),
            sublinear_tf=True,
            lowercase=True,
        )
        documents = [self._document(faq) for faq in faqs]
        self.matrix = self.vectorizer.fit_transform(documents)

    @staticmethod
    def _document(faq: dict[str, Any]) -> str:
        return " ".join([faq["question"], *faq.get("keywords", [])])

    def search(self, question: str, limit: int = 3) -> list[Match]:
        if not question.strip():
            return []
        query_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        order = sorted(range(len(self.faqs)), key=lambda i: (-float(scores[i]), self.faqs[i]["id"]))
        return [
            Match(self.faqs[i]["id"], self.faqs[i]["answer"], float(scores[i]))
            for i in order[: max(1, limit)]
        ]

    def answer(self, question: str) -> Match | None:
        matches = self.search(question, limit=1)
        if not matches or matches[0].score < self.threshold:
            return None
        return matches[0]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_queries(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
