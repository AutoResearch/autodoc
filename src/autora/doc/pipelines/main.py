import itertools
import logging
from timeit import default_timer as timer
from typing import List, Tuple

import nltk
import torch
import typer
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import single_meteor_score

from autora.doc.runtime.predict_hf import Predictor
from autora.doc.runtime.prompts import PROMPTS, PromptIds

app = typer.Typer()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(module)s.%(funcName)s(): %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate_documentation(predictions: List[List[str]], references: List[str]) -> Tuple[float, float]:
    nltk.download("wordnet")

    # Tokenize references
    tokenized_references = [ref.split() for ref in references]
    # Currently there is only 1 prediction for 1 reference, need to avg in future
    tokenized_predictions = [pred[0].split() if pred else [] for pred in predictions]

    # Calculate BLEU score with smoothing function
    # SmoothingFunction().method1 is used to avoid zero scores for n-grams not found in the reference.
    bleu = corpus_bleu(
        # Wrap each reference list in another list
        [[tokenized_ref] for tokenized_ref in tokenized_references],
        tokenized_predictions,
        smoothing_function=SmoothingFunction().method1,
    )

    # Calculate METEOR scores
    meteor_scores = [
        single_meteor_score(tokenized_ref, tokenized_pred)
        for tokenized_ref, tokenized_pred in zip(tokenized_references, tokenized_predictions)
    ]
    meteor = sum(meteor_scores) / len(predictions) if predictions else 0

    return (bleu, meteor)


@app.command(help="Evaluate model on a data file")
def eval(
    data_file: str = typer.Argument(..., help="JSONL Data file to evaluate on"),
    model_path: str = typer.Option("meta-llama/Llama-2-7b-chat-hf", help="Path to HF model"),
    prompt_id: PromptIds = typer.Option(PromptIds.SWEETP_1, help="Instruction prompt ID"),
    param: List[str] = typer.Option(
        [], help="Additional float parameters to pass to the model as name=float pairs"
    ),
) -> List[List[str]]:
    import jsonlines
    import mlflow

    mlflow.autolog()

    param_dict = {pair[0]: float(pair[1]) for pair in [pair.split("=") for pair in param]}
    run = mlflow.active_run()

    prompt = PROMPTS[prompt_id]
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

        with jsonlines.open(data_file) as reader:
            items = [item for item in reader]
            inputs = [item["instruction"] for item in items]
            labels = [item["output"] for item in items]

        pred = Predictor(model_path)
        timer_start = timer()
        predictions = pred.predict(prompt, inputs, **param_dict)
        timer_end = timer()
        bleu, meteor = evaluate_documentation(predictions, labels)
        pred_time = timer_end - timer_start
        mlflow.log_metric("prediction_time/doc", pred_time / (len(inputs)))
        for i in range(len(inputs)):
            mlflow.log_text(labels[i], f"label_{i}.txt")
            mlflow.log_text(inputs[i], f"input_{i}.py")
            for j in range(len(predictions[i])):
                mlflow.log_text(predictions[i][j], f"prediction_{i}_{j}.txt")
        mlflow.log_text("bleu_score is ", str(bleu))
        mlflow.log_text("meteor_score is ", str(meteor))

        # flatten predictions for counting tokens
        predictions_flat = list(itertools.chain.from_iterable(predictions))
        tokens = pred.tokenize(predictions_flat)["input_ids"]
        total_tokens = sum([len(token) for token in tokens])
        mlflow.log_metric("total_tokens", total_tokens)
        mlflow.log_metric("tokens/sec", total_tokens / pred_time)
        mlflow.log_metric("bleu_score", round(bleu, 5))
        mlflow.log_metric("meteor_score", round(meteor, 5))
        return predictions


@app.command()
def generate(
    python_file: str = typer.Argument(..., help="Python file to generate documentation for"),
    model_path: str = typer.Option("meta-llama/Llama-2-7b-chat-hf", help="Path to HF model"),
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
    predictions = pred.predict(prompt, [input], **param_dict)[0]
    assert len(predictions) == 1, f"Expected only one output, got {len(predictions)}"
    logger.info(f"Writing output to {output}")
    with open(output, "w") as f:
        f.write(predictions[0])


@app.command()
def import_model(model_name: str) -> None:
    pass


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
    logger.info(f"Torch version: {torch.__version__} , Cuda available: {torch.cuda.is_available()}")

    app()
