from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from .retrieval import FAQRetriever, load_json, load_queries

ROOT = Path(__file__).resolve().parents[2]


def evaluate_many(faqs, queries, analyzer: str, thresholds: list[float]) -> list[dict]:
    """Evaluate several abstention cutoffs while ranking each query only once."""
    if not thresholds:
        raise ValueError("at least one threshold is required")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("thresholds must be finite values between 0 and 1")

    retriever = FAQRetriever(faqs, analyzer=analyzer, threshold=thresholds[0])
    answerable = [row for row in queries if row["expected_faq_id"] is not None]
    non_answerable = [row for row in queries if row["expected_faq_id"] is None]
    ranked_queries = []
    latencies = []
    for row in queries:
        start = time.perf_counter()
        ranked = retriever.search(row["question"], limit=3)
        latencies.append((time.perf_counter() - start) * 1000)
        ranked_queries.append((row, ranked))

    median_latency = round(statistics.median(latencies), 3) if latencies else None
    results = []
    for threshold in thresholds:
        top1_hits = 0
        recall_hits = 0
        total_answered = 0
        unsupported_answers = 0
        non_answerable_abstentions = 0
        for row, ranked in ranked_queries:
            chosen = ranked[0] if ranked and ranked[0].score >= threshold else None
            expected = row["expected_faq_id"]
            if expected is not None:
                top1_hits += int(chosen is not None and chosen.faq_id == expected)
                recall_hits += int(any(match.faq_id == expected for match in ranked))
            if chosen is None and expected is None:
                non_answerable_abstentions += 1
            if chosen is not None:
                total_answered += 1
                unsupported_answers += int(expected is None)

        results.append({
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
            "median_query_latency_ms": median_latency,
        })
    return results


def evaluate(faqs, queries, analyzer: str, threshold: float) -> dict:
    return evaluate_many(faqs, queries, analyzer, [threshold])[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ClearAsk's synthetic FAQ retriever.")
    parser.add_argument("--threshold", type=float, default=0.22)
    parser.add_argument("--sweep", type=float, nargs="+", metavar="THRESHOLD",
                        help="evaluate multiple abstention thresholds in one run")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    faqs = load_json(ROOT / "data" / "faqs.json")
    queries = load_queries(ROOT / "data" / "queries.jsonl")
    report = {
        "dataset": "synthetic campus-service FAQ benchmark",
        "note": "Scores are a small synthetic benchmark, not evidence of real-world service performance.",
    }
    if args.sweep is not None:
        report["threshold_sweep"] = {
            analyzer: evaluate_many(faqs, queries, analyzer, args.sweep)
            for analyzer in ("word", "char")
        }
    else:
        report["results"] = [
            evaluate(faqs, queries, analyzer, args.threshold) for analyzer in ("word", "char")
        ]
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
