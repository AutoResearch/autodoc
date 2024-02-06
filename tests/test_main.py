from pathlib import Path
from typing import List

from autora.doc.classes.EvalResult import EvalResult
from autora.doc.pipelines.main import eval, eval_prompts, generate, import_data
from autora.doc.runtime.prompts import PromptIds

# dummy HF model for testing
TEST_HF_MODEL = "hf-internal-testing/tiny-random-FalconForCausalLM"


def test_predict() -> None:
    data = Path(__file__).parent.joinpath("../data/sweetpea/data.jsonl").resolve()
    outputs, _, _ = eval(str(data), TEST_HF_MODEL, PromptIds.SWEETP_1, [])
    assert len(outputs) == 3, "Expected 3 outputs"
    for output in outputs:
        assert len(output) > 0, "Expected non-empty output"


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


def test_eval_prompts() -> None:
    data_file = Path(__file__).parent.joinpath("../data/sweetpea/data.jsonl").resolve()
    prompts_file = Path(__file__).parent.joinpath("../data/autora/prompts/all_prompt.json").resolve()
    results: List[EvalResult] = eval_prompts(str(data_file), TEST_HF_MODEL, str(prompts_file), [])
    assert len(results) == 3, "Expected 3 outputs"
    for result in results:
        assert result.prediction is not None, "The prediction should not be None"
        assert result.prompt is not None, "The prompt should not be None"
