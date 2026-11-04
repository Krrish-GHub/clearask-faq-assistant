# ClearAsk: FAQ retrieval under everyday phrasing

A small, reproducible AI/ML experiment: can a simple FAQ retriever find the right service answer when a question is indirect, misspelled, or underspecified—and when should it abstain?

## Why publish it?

People ask the same question in many ways. A retriever that matches only formal wording can leave someone stuck; a retriever that always guesses can give a confidently wrong answer. This project makes both behavior and failure cases testable with synthetic examples.

The campus-service FAQs and queries are invented for this benchmark. They contain no real student records, private messages, or personal data. They are not official guidance for any real institution.

## What it tests

- A word n-gram TF-IDF baseline and a character n-gram TF-IDF baseline.
- Direct, paraphrased, typo-containing, and indirect queries.
- Ambiguous and out-of-scope queries, where the system should abstain below a similarity threshold.
- Top-1 accuracy, recall@3, coverage, unsupported-answer rate, abstention recall, and median query latency.

The retriever returns an existing FAQ answer; it does not generate facts or answers. The threshold is a simple configurable score cutoff, not a calibrated confidence estimate.

## First synthetic baseline

The initial 30-query benchmark (24 answerable, 6 expected abstentions) produced these results at threshold `0.22`, using Python 3.14.7 and scikit-learn 1.9.1:

| Analyzer | Top-1 accuracy | Recall@3 | Coverage | Unsupported-answer rate | Abstention recall | Median latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Word n-grams | 0.542 | 0.917 | 0.700 | 0.190 | 0.333 | 0.564 ms |
| Character n-grams | 0.708 | 0.875 | 0.733 | 0.136 | 0.500 | 0.598 ms |

On this small authored set, character n-grams ranked the expected FAQ first more often and made fewer unsupported matches, but the benchmark is too small and synthetic to claim real-world superiority. These numbers are a baseline for improving the dataset and evaluation, not a product-quality claim. Exact output is saved in `results/baseline.json`.

## Run it

```bash
uv sync --extra dev
uv run pytest
uv run clearask-evaluate
```

To change the abstention cutoff or save a local JSON report:

```bash
uv run clearask-evaluate --threshold 0.22 --output results/report.json
```

## Reproducibility and data

- `data/faqs.json` is the complete synthetic knowledge base.
- `data/queries.jsonl` is the complete, labelled evaluation set. Each query has a category and either an expected FAQ id or `null` when it should be rejected.
- `uv.lock` pins the dependency graph used for the checked-in baseline.
- No network access or external dataset is needed to run tests or the benchmark.
- The evaluator reports results by analyzer. Run it on the same machine and dependency versions when comparing revisions; latency is machine-dependent.

## Limitations

This is a teaching-scale benchmark written to make an undergrad-level retrieval experiment inspectable, not a production assistant or a claim about real people. The synthetic set is small and authored by one person, so its wording and difficulty may be biased. Similarity scores are not calibrated probabilities; a threshold can trade coverage for fewer unsupported matches. The labelled test questions also share the same synthetic domain and FAQ bank, so results do not establish generalization to real services, other institutions, languages, or users. No real-person testing has been done.

## A defensible project explanation

> I published ClearAsk to test whether simple text-retrieval methods handle everyday variation in service questions, and to make the cases where a system should refuse to guess visible. I used invented data so the experiment is reproducible without collecting anyone's private information.

## Status

The first TF-IDF baselines, a synthetic test set, and four unit tests are in place. The next useful experiment is a confidence-threshold sweep on ambiguous and out-of-scope queries; this small initial benchmark is not a finished or validated real-world system.
