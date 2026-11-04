import json
from pathlib import Path

from clearask.retrieval import FAQRetriever, load_json

ROOT = Path(__file__).resolve().parents[1]


def test_exact_question_finds_its_faq():
    faqs = load_json(ROOT / "data" / "faqs.json")
    retriever = FAQRetriever(faqs)
    assert retriever.answer("What are the university library opening hours?").faq_id == "library-hours"


def test_empty_or_unrelated_question_abstains_at_high_threshold():
    faqs = load_json(ROOT / "data" / "faqs.json")
    retriever = FAQRetriever(faqs, threshold=1.0)
    assert retriever.answer("") is None
    assert retriever.answer("How do I appeal a parking fine?") is None


def test_search_ranks_stably_and_respects_limit():
    faqs = load_json(ROOT / "data" / "faqs.json")
    retriever = FAQRetriever(faqs)
    first = retriever.search("How can I book a study room?", limit=3)
    second = retriever.search("How can I book a study room?", limit=3)
    assert len(first) == 3
    assert [match.faq_id for match in first] == [match.faq_id for match in second]
    assert first[0].faq_id == "study-room"


def test_rejects_invalid_configuration():
    faqs = load_json(ROOT / "data" / "faqs.json")
    try:
        FAQRetriever(faqs, analyzer="semantic")
    except ValueError as error:
        assert "analyzer" in str(error)
    else:
        raise AssertionError("invalid analyzer should be rejected")
