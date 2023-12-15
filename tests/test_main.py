from pathlib import Path

import jsonlines

from autora.doc.pipelines.main import eval, generate, evaluate_documentation
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
    # Test Case: Valid Scores in the range of 0 and 1
    data = Path(__file__).parent.joinpath("../data/data.jsonl").resolve()
    with jsonlines.open(data) as reader:
            items = [item for item in reader]
            labels = [item["output"] for item in items]
    
    bleu, meteor = evaluate_documentation(labels, labels)
    assert bleu >= 0 and bleu <= 1, "BLEU score should be between 0 and 1"
    assert meteor >= 0 and meteor <= 1, "METEOR score should be between 0 and 1"

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
