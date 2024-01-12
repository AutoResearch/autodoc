from pathlib import Path

import jsonlines
import pytest

from autora.doc.pipelines.main import eval, evaluate_documentation, generate
from autora.doc.runtime.prompts import InstructionPrompts, SystemPrompts

# dummy HF model for testing
TEST_HF_MODEL = "hf-internal-testing/tiny-random-FalconForCausalLM"


def test_predict() -> None:
    data = Path(__file__).parent.joinpath("../data/data.jsonl").resolve()
    outputs = eval(str(data), TEST_HF_MODEL, SystemPrompts.SYS_1, InstructionPrompts.INSTR_SWEETP_1, [])
    assert len(outputs) == 3, "Expected 3 outputs"
    for output in outputs:
        assert len(output[0]) > 0, "Expected non-empty output"


def test_evaluation() -> None:
    # Test Case: Meteor and Bleu scores are close to 1
    data = Path(__file__).parent.joinpath("../data/data.jsonl").resolve()
    with jsonlines.open(data) as reader:
        items = [item for item in reader]
        labels = [item["output"] for item in items]
        predictions = [[item["output"]] for item in items]

    bleu, meteor = evaluate_documentation(predictions, labels)
    assert bleu == pytest.approx(1, 0.01), f"BLEU Score is {bleu}"
    assert meteor == pytest.approx(1, 0.01), f"METEOR Score is {meteor}"


def test_generate() -> None:
    python_file = __file__
    output = Path("output.txt")
    output.unlink(missing_ok=True)
    generate(
        python_file, TEST_HF_MODEL, str(output), SystemPrompts.SYS_1, InstructionPrompts.INSTR_SWEETP_1, []
    )
    assert output.exists(), f"Expected output file {output} to exist"
    with open(str(output), "r") as f:
        assert len(f.read()) > 0, f"Expected non-empty output file {output}"
