import itertools
import logging
from timeit import default_timer as timer
from typing import Dict, List

import torch
import typer

from autora.doc.classes.EvalResult import EvalResult
from autora.doc.pipelines.data import load_data
from autora.doc.pipelines.metrics import eval_bleu_meteor, eval_semscore
from autora.doc.pipelines.train import fine_tune, get_dataset
from autora.doc.runtime.predict_hf import Predictor
from autora.doc.runtime.prompts import PROMPTS, PromptIds
from autora.doc.util import get_prompts_from_file

# For inference
DEFAULT_INFERENCE_MODEL = "meta-llama/Llama-2-7b-chat-hf"
# For training
DEFAULT_BASE_MODEL = "autora-doc/Llama-2-7b-chat-hf"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(module)s.%(funcName)s(): %(message)s",
)
logger = logging.getLogger(__name__)
logger.info(f"Torch version: {torch.__version__} , Cuda available: {torch.cuda.is_available()}")
app = typer.Typer()


@app.command(help="Evaluate a model for code-to-documentation generation for all prompts in the prompts_file")
def eval_prompts(
    data_file: str = typer.Argument(..., help="JSONL Data file to evaluate on"),
    model_path: str = typer.Option(DEFAULT_INFERENCE_MODEL, help="Path to HF model"),
    prompts_file: str = typer.Argument(..., help="JSON file with a list of dictionary of prompts"),
    param: List[str] = typer.Option(
        [], help="Additional float parameters to pass to the model as name=float pairs"
    ),
) -> List[EvalResult]:
    import mlflow

    results_list = []

    mlflow.autolog()
    param_dict = {pair[0]: float(pair[1]) for pair in [pair.split("=") for pair in param]}
    run = mlflow.active_run()

    prompts_list = get_prompts_from_file(prompts_file)

    if run is None:
        run = mlflow.start_run()
    with run:
        logger.info(f"Active run_id: {run.info.run_id}")
        logger.info(f"running predict with {data_file}")
        logger.info(f"model path: {model_path}")
        mlflow.log_params(param_dict)
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("data_file", data_file)
        mlflow.log_param("prompts_file", prompts_file)
        predictor = Predictor(model_path)
        for i in range(len(prompts_list)):
            logger.info(f"Starting to run model on prompt {i}")
            eval_result = eval_prompt(data_file, predictor, prompts_list[i], param_dict, i)
            logger.info(f"Model run completed on prompt {i}: {prompts_list[i]}")
            results_list.append(eval_result)
        return results_list


@app.command(help="Evaluate model on a data file")
def eval(
    data_file: str = typer.Argument(..., help="JSONL Data file to evaluate on"),
    model_path: str = typer.Option(DEFAULT_INFERENCE_MODEL, help="Path to HF model"),
    prompt_id: PromptIds = typer.Option(PromptIds.SWEETP_1, help="Instruction prompt ID"),
    param: List[str] = typer.Option(
        [], help="Additional float parameters to pass to the model as name=float pairs"
    ),
) -> EvalResult:
    import mlflow

    mlflow.autolog()
    run = mlflow.active_run()
    param_dict = {pair[0]: float(pair[1]) for pair in [pair.split("=") for pair in param]}

    if run is None:
        run = mlflow.start_run()
    with run:
        logger.info(f"Active run_id: {run.info.run_id}")
        logger.info(f"running predict with {data_file}")
        logger.info(f"model path: {model_path}")
        mlflow.log_params(param_dict)
        mlflow.log_param("prompt_id", prompt_id)
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("data_file", data_file)
        prompt = PROMPTS[prompt_id]
        pred = Predictor(model_path)
        return eval_prompt(data_file, pred, prompt, param_dict)


def eval_prompt(
    data_file: str, pred: Predictor, prompt: str, param_dict: Dict[str, float], prompt_index: int = 0
) -> EvalResult:
    import mlflow

    inputs, labels = load_data(data_file)

    timer_start = timer()
    predictions = pred.predict(prompt, inputs, **param_dict)
    timer_end = timer()
    bleu, meteor = eval_bleu_meteor(predictions, labels)
    semscore = eval_semscore(predictions, labels)
    pred_time = timer_end - timer_start
    mlflow.log_metric("prediction_time/doc", pred_time / (len(inputs)))
    for i in range(len(inputs)):
        mlflow.log_text(predictions[i], f"prompt_{prompt_index}_prediction_{i}.txt")

    # flatten predictions for counting tokens
    predictions_flat = list(itertools.chain.from_iterable(predictions))
    tokens = pred.tokenize(predictions_flat)["input_ids"]
    total_tokens = sum([len(token) for token in tokens])
    metrics_dict = {
        f"prompt_{prompt_index}_total_tokens": total_tokens,
        f"prompt_{prompt_index}_tokens/sec": total_tokens / pred_time,
        f"prompt_{prompt_index}_bleu_score": round(bleu, 5),
        f"prompt_{prompt_index}_meteor_score": round(meteor, 5),
        f"prompt_{prompt_index}_semscore": round(semscore, 5),
    }
    mlflow.log_metrics(metrics_dict)
    return EvalResult(predictions, prompt, bleu, meteor, semscore)


@app.command()
def generate(
    python_file: str = typer.Argument(..., help="Python file to generate documentation for"),
    model_path: str = typer.Option(DEFAULT_INFERENCE_MODEL, help="Path to HF model"),
    output: str = typer.Option("output.txt", help="Output file"),
    prompt_id: PromptIds = typer.Option(PromptIds.SWEETP_1, help="Instruction prompt ID"),
    param: List[str] = typer.Option(
        [], help="Additional float parameters to pass to the model as name=float pairs"
    ),
) -> None:
    param_dict = {pair[0]: float(pair[1]) for pair in [pair.split("=") for pair in param]}
    """
    Generate documentation from python file
    """
    with open(python_file, "r") as f:
        input = f.read()
    prompt = PROMPTS[prompt_id]
    pred = Predictor(model_path)
    # grab first result since we only passed one input
    predictions = pred.predict(prompt, [input], **param_dict)
    assert len(predictions) == 1, f"Expected only one output, got {len(predictions)}"
    logger.info(f"Writing output to {output}")
    with open(output, "w", encoding="utf-8") as f:
        f.write(predictions[0])


@app.command()
def import_model(model_name: str) -> None:
    pass


@app.command()
def train(
    new_model_name: str = typer.Argument(..., help="File name for the fine-tuned model"),
    dataset: str = typer.Argument(..., help="Path to the jsonl file with training data"),
    base_model: str = typer.Option(
        DEFAULT_BASE_MODEL, help="Path to the base Huggingface model to fine-tune"
    ),
) -> None:
    ds = get_dataset(dataset)
    fine_tune(base_model, new_model_name, ds)


@app.command()
def import_data(code_file: str, text_file: str, output_file: str = "data.jsonl") -> None:
    from pathlib import Path

    import jsonlines

    # alpaca jsonl format:
    def read_text(file: str) -> str:
        return Path(file).read_text()

    d = {"instruction": read_text(code_file), "output": read_text(text_file)}
    with jsonlines.open(output_file, "a") as file:
        file.write(d)


if __name__ == "__main__":
    app()
