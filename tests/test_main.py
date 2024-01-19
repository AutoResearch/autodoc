from pathlib import Path

import jsonlines
import pytest

from autora.doc.pipelines.main import eval, evaluate_documentation, generate, import_data
from autora.doc.runtime.prompts import PromptIds

# dummy HF model for testing
TEST_HF_MODEL = "hf-internal-testing/tiny-random-FalconForCausalLM"


def test_predict() -> None:
    data = Path(__file__).parent.joinpath("../data/sweetpea/data.jsonl").resolve()
    outputs = eval(str(data), TEST_HF_MODEL, PromptIds.SWEETP_1, [])
    assert len(outputs) == 3, "Expected 3 outputs"
    for output in outputs:
        assert len(output[0]) > 0, "Expected non-empty output"


def test_evaluation() -> None:
    # Test Case: Meteor and Bleu scores are close to 1
    data = Path(__file__).parent.joinpath("../data/sweetpea/data.jsonl").resolve()
    with jsonlines.open(data) as reader:
        items = [item for item in reader]
        labels = [item["output"] for item in items]
        predictions = [[item["output"]] for item in items]

    bleu, meteor = evaluate_documentation(predictions, labels)
    assert bleu == pytest.approx(1, 0.01), f"BLEU Score is {bleu}"
    assert meteor == pytest.approx(1, 0.01), f"METEOR Score is {meteor}"


def test_extra_token_in_prediction() -> None:
    # Test Case bleu score should be less due to brevity penalty and meteor is robust to small mistakes
    labels = ["this is a test"]
    predictions = [["this is a test extra"]]
    bleu, meteor = evaluate_documentation(predictions, labels)
    assert 0.6 <= bleu <= 0.8, f"BLEU Score is {bleu}"
    assert 0.8 <= meteor <= 1, f"METEOR Score is {meteor}"


def test_missing_token_in_prediction() -> None:
    # bleu score is less, meteor is higher
    labels = ["this is a test"]
    predictions = [["this is a"]]
    bleu, meteor = evaluate_documentation(predictions, labels)
    assert 0.4 <= bleu <= 0.6, f"BLEU Score is {bleu}"
    assert 0.6 <= meteor <= 0.8, f"METEOR Score is {meteor}"


def test_completely_different_tokens() -> None:
    # both scores are less, as no common tokens
    labels = ["this is a test"]
    predictions = [["completely different sentence"]]
    bleu, meteor = evaluate_documentation(predictions, labels)
    assert bleu <= 0.1, f"BLEU Score is {bleu}"
    assert meteor <= 0.1, f"METEOR Score is {meteor}"


def test_partially_matching_tokens() -> None:
    # As ngrams arent matching because of extra token within, BLEU score is very less. Meteor gives a good score only.
    labels = ["this is a test"]
    predictions = [["this is a different test"]]
    bleu, meteor = evaluate_documentation(predictions, labels)
    assert 0.25 <= bleu <= 0.4, f"BLEU Score is {bleu}"
    assert 0.8 <= meteor <= 0.95, f"METEOR Score is {meteor}"


def test_generate() -> None:
    python_file = __file__
    output = Path("output.txt")
    output.unlink(missing_ok=True)
    generate(python_file, TEST_HF_MODEL, str(output), PromptIds.SWEETP_1, [])
    assert output.exists(), f"Expected output file {output} to exist"
    with open(str(output), "r") as f:
        assert len(f.read()) > 0, f"Expected non-empty output file {output}"


def test_import(tmp_path: Path) -> None:
    data = tmp_path.joinpath("data.jsonl")
    code = Path(__file__).parent.joinpath("../data/autora/code1.txt").resolve()
    text = Path(__file__).parent.joinpath("../data/autora/text1.txt").resolve()
    import_data(str(code), str(text), str(data))
    new_lines = data.read_text().splitlines()
    assert len(new_lines) == 1, "Expected one new line"
