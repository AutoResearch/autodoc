from pathlib import Path

import jsonlines
import pytest

from autora.doc.pipelines.metrics import eval_bleu_meteor, eval_semscore


def test_evaluation() -> None:
    # Test Case: Meteor and Bleu scores are close to 1
    data = Path(__file__).parent.joinpath("../data/sweetpea/data.jsonl").resolve()
    with jsonlines.open(data) as reader:
        items = [item for item in reader]
        labels = [item["output"] for item in items]
        predictions = [item["output"] for item in items]

    bleu, meteor = eval_bleu_meteor(predictions, labels)
    assert bleu == pytest.approx(1, 0.01), f"BLEU Score is {bleu}"
    assert meteor == pytest.approx(1, 0.01), f"METEOR Score is {meteor}"


def test_extra_token_in_prediction() -> None:
    # Test Case bleu score should be less due to brevity penalty and meteor is robust to small mistakes
    labels = ["this is a test"]
    predictions = ["this is a test extra"]
    bleu, meteor = eval_bleu_meteor(predictions, labels)
    assert 0.6 <= bleu <= 0.8, f"BLEU Score is {bleu}"
    assert 0.8 <= meteor <= 1, f"METEOR Score is {meteor}"


def test_missing_token_in_prediction() -> None:
    # bleu score is less, meteor is higher
    labels = ["this is a test"]
    predictions = ["this is a"]
    bleu, meteor = eval_bleu_meteor(predictions, labels)
    assert 0.4 <= bleu <= 0.6, f"BLEU Score is {bleu}"
    assert 0.6 <= meteor <= 0.8, f"METEOR Score is {meteor}"


def test_completely_different_tokens() -> None:
    # both scores are less, as no common tokens
    labels = ["this is a test"]
    predictions = ["completely different sentence"]
    bleu, meteor = eval_bleu_meteor(predictions, labels)
    assert bleu <= 0.1, f"BLEU Score is {bleu}"
    assert meteor <= 0.1, f"METEOR Score is {meteor}"


def test_partially_matching_tokens() -> None:
    # As ngrams arent matching because of extra token within, BLEU score is very less. Meteor gives a good score only.
    labels = ["this is a test"]
    predictions = ["this is a different test"]
    bleu, meteor = eval_bleu_meteor(predictions, labels)
    assert 0.25 <= bleu <= 0.4, f"BLEU Score is {bleu}"
    assert 0.8 <= meteor <= 0.95, f"METEOR Score is {meteor}"


def test_semscore() -> None:
    # Test Case: SemScore is close to 1
    labels = ["this is really good"]
    predictions = ["this is great"]
    semscore = eval_semscore(predictions, labels)
    assert semscore >= 0.6, f"SemScore is {semscore}"

    semscore = eval_semscore(labels, labels)
    assert semscore == pytest.approx(1.0), f"SemScore is {semscore}"

    semscore = eval_semscore([], [])
    assert semscore == 0, f"SemScore is {semscore}"
