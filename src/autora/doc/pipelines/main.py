import itertools
import logging
from timeit import default_timer as timer
from typing import List

import torch
import typer

from autora.doc.runtime.predict_hf import Predictor
from autora.doc.runtime.prompts import INSTR, SYS, InstructionPrompts, SystemPrompts

app = typer.Typer()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(module)s.%(funcName)s(): %(message)s",
)
logger = logging.getLogger(__name__)


@app.command(help="Evaluate model on a data file")
def eval(
    data_file: str = typer.Argument(..., help="JSONL Data file to evaluate on"),
    model_path: str = typer.Option("meta-llama/Llama-2-7b-chat-hf", help="Path to HF model"),
    sys_id: SystemPrompts = typer.Option(SystemPrompts.SYS_1, help="System prompt ID"),
    instruc_id: InstructionPrompts = typer.Option(
        InstructionPrompts.INSTR_SWEETP_1, help="Instruction prompt ID"
    ),
    param: List[str] = typer.Option(
        [], help="Additional float parameters to pass to the model as name=float pairs"
    ),
) -> List[List[str]]:
    import jsonlines
    import mlflow

    mlflow.autolog()

    param_dict = {pair[0]: float(pair[1]) for pair in [pair.split("=") for pair in param]}
    run = mlflow.active_run()

    sys_prompt = SYS[sys_id]
    instr_prompt = INSTR[instruc_id]
    if run is None:
        run = mlflow.start_run()
    with run:
        logger.info(f"Active run_id: {run.info.run_id}")
        logger.info(f"running predict with {data_file}")
        logger.info(f"model path: {model_path}")
        mlflow.log_params(param_dict)

        with jsonlines.open(data_file) as reader:
            items = [item for item in reader]
            inputs = [item["instruction"] for item in items]
            labels = [item["output"] for item in items]

        pred = Predictor(model_path)
        timer_start = timer()
        predictions = pred.predict(sys_prompt, instr_prompt, inputs, **param_dict)
        timer_end = timer()
        pred_time = timer_end - timer_start
        mlflow.log_metric("prediction_time/doc", pred_time / (len(inputs)))
        for i in range(len(inputs)):
            mlflow.log_text(labels[i], f"label_{i}.txt")
            mlflow.log_text(inputs[i], f"input_{i}.py")
            for j in range(len(predictions[i])):
                mlflow.log_text(predictions[i][j], f"prediction_{i}_{j}.txt")

        # flatten predictions for counting tokens
        predictions_flat = list(itertools.chain.from_iterable(predictions))
        tokens = pred.tokenize(predictions_flat)["input_ids"]
        total_tokens = sum([len(token) for token in tokens])
        mlflow.log_metric("total_tokens", total_tokens)
        mlflow.log_metric("tokens/sec", total_tokens / pred_time)
        return predictions


@app.command()
def generate(
    python_file: str = typer.Argument(..., help="Python file to generate documentation for"),
    model_path: str = typer.Option("meta-llama/Llama-2-7b-chat-hf", help="Path to HF model"),
    output: str = typer.Option("output.txt", help="Output file"),
    sys_id: SystemPrompts = typer.Option(SystemPrompts.SYS_1, help="System prompt ID"),
    instruc_id: InstructionPrompts = typer.Option(
        InstructionPrompts.INSTR_SWEETP_1, help="Instruction prompt ID"
    ),
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
    sys_prompt = SYS[sys_id]
    instr_prompt = INSTR[instruc_id]
    pred = Predictor(model_path)
    # grab first result since we only passed one input
    predictions = pred.predict(sys_prompt, instr_prompt, [input], **param_dict)[0]
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
