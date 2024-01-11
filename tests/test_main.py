from pathlib import Path

from autora.doc.pipelines.main import eval, generate
from autora.doc.runtime.prompts import InstructionPrompts, SystemPrompts

# dummy HF model for testing
TEST_HF_MODEL = "hf-internal-testing/tiny-random-FalconForCausalLM"


def test_predict() -> None:
    data = Path(__file__).parent.joinpath("../data/sweetpea/data.jsonl").resolve()
    outputs = eval(str(data), TEST_HF_MODEL, SystemPrompts.SYS_1, InstructionPrompts.INSTR_SWEETP_1, [])
    assert len(outputs) == 3, "Expected 3 outputs"
    for output in outputs:
        assert len(output[0]) > 0, "Expected non-empty output"


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
