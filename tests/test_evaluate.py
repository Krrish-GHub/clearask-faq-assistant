from pathlib import Path

import pytest

from clearask.evaluate import evaluate, evaluate_many
from clearask.retrieval import load_json, load_queries

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("analyzer", ["word", "char"])
def test_threshold_sweep_shares_metrics_and_trades_coverage_for_abstention(analyzer):
    faqs = load_json(ROOT / "data" / "faqs.json")
    queries = load_queries(ROOT / "data" / "queries.jsonl")
    thresholds = [0.0, 0.22, 0.5, 1.0]

    sweep = evaluate_many(faqs, queries, analyzer, thresholds)

    assert [row["threshold"] for row in sweep] == thresholds
    single_run = evaluate(faqs, queries, analyzer, 0.22)
    assert {key: value for key, value in sweep[1].items() if key != "median_query_latency_ms"} == {
        key: value for key, value in single_run.items() if key != "median_query_latency_ms"
    }
    assert [row["coverage"] for row in sweep] == sorted(
        (row["coverage"] for row in sweep), reverse=True
    )
    assert [row["abstention_recall_on_ambiguous_and_out_of_scope"] for row in sweep] == sorted(
        row["abstention_recall_on_ambiguous_and_out_of_scope"] for row in sweep
    )


def test_threshold_sweep_rejects_empty_or_out_of_range_thresholds():
    faqs = load_json(ROOT / "data" / "faqs.json")
    queries = load_queries(ROOT / "data" / "queries.jsonl")

    with pytest.raises(ValueError, match="at least one threshold"):
        evaluate_many(faqs, queries, "word", [])
    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluate_many(faqs, queries, "word", [0.2, 1.1])
