from unittest import mock

from autora.doc.runtime.predict_hf import Predictor, quantized_models

# Test models with and without available quantized models
MODEL_NO_QUANTIZED = "hf-internal-testing/tiny-random-FalconForCausalLM"
MODEL_WITH_QUANTIZED = "meta-llama/Llama-2-7b-chat-hf"


def test_trim_prompt() -> None:
    """Verify the output of the `trim_prompt` function"""
    no_marker = "Generated text with no marker"
    output = Predictor.trim_prompt(no_marker)
    assert output == no_marker

    with_marker = """
The prompt is here
[/INST]
output
"""
    output = Predictor.trim_prompt(with_marker)
    assert output == "output\n"


@mock.patch("torch.cuda.is_available", return_value=True)
def test_get_config_cuda(mock: mock.Mock) -> None:
    model, config = Predictor.get_config(MODEL_WITH_QUANTIZED)
    assert model == quantized_models[MODEL_WITH_QUANTIZED]
    assert "quantization_config" not in config

    model, config = Predictor.get_config(MODEL_NO_QUANTIZED)
    # no pre-quantized model available
    assert model == MODEL_NO_QUANTIZED
    assert "quantization_config" in config


@mock.patch("torch.cuda.is_available", return_value=False)
def test_get_config_nocuda(mock: mock.Mock) -> None:
    model, config = Predictor.get_config(MODEL_NO_QUANTIZED)
    assert model == MODEL_NO_QUANTIZED
    assert len(config) == 0
