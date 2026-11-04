from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from .retrieval import FAQRetriever, load_json, load_queries

ROOT = Path(__file__).resolve().parents[2]


def evaluate(faqs, queries, analyzer: str, threshold: float) -> dict:
    retriever = FAQRetriever(faqs, analyzer=analyzer, threshold=threshold)
    answerable = [row for row in queries if row["expected_faq_id"] is not None]
    non_answerable = [row for row in queries if row["expected_faq_id"] is None]
    top1_hits = 0
    recall_hits = 0
    total_answered = 0
    unsupported_answers = 0
    non_answerable_abstentions = 0
    latencies = []

    for row in queries:
        start = time.perf_counter()
        ranked = retriever.search(row["question"], limit=3)
        chosen = ranked[0] if ranked and ranked[0].score >= threshold else None
        latencies.append((time.perf_counter() - start) * 1000)
        expected = row["expected_faq_id"]
        if expected is not None:
            top1_hits += int(chosen is not None and chosen.faq_id == expected)
            recall_hits += int(any(match.faq_id == expected for match in ranked))
        if chosen is None and expected is None:
            non_answerable_abstentions += 1
        if chosen is not None:
            total_answered += 1
            unsupported_answers += int(expected is None)

    return {
        "analyzer": analyzer,
        "threshold": threshold,
        "queries": len(queries),
        "answerable_queries": len(answerable),
        "top1_accuracy": round(top1_hits / len(answerable), 3) if answerable else None,
        "recall_at_3": round(recall_hits / len(answerable), 3) if answerable else None,
        "coverage": round(total_answered / len(queries), 3) if queries else None,
        "unsupported_answer_rate": round(unsupported_answers / total_answered, 3) if total_answered else 0.0,
        "abstention_recall_on_ambiguous_and_out_of_scope": round(
            non_answerable_abstentions / len(non_answerable), 3
        ) if non_answerable else None,
        "median_query_latency_ms": round(statistics.median(latencies), 3) if latencies else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ClearAsk's synthetic FAQ retriever.")
    parser.add_argument("--threshold", type=float, default=0.22)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    faqs = load_json(ROOT / "data" / "faqs.json")
    queries = load_queries(ROOT / "data" / "queries.jsonl")
    report = {
        "dataset": "synthetic campus-service FAQ benchmark",
        "note": "Scores are a small synthetic benchmark, not evidence of real-world service performance.",
        "results": [
            evaluate(faqs, queries, analyzer, args.threshold) for analyzer in ("word", "char")
        ],
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
