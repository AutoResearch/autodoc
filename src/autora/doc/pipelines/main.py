import logging
from timeit import default_timer as timer

import jsonlines
import mlflow
import torch
import typer

from autora.doc.runtime.predict_hf import Predictor

app = typer.Typer()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(module)s.%(funcName)s(): %(message)s",
)
logger = logging.getLogger(__name__)

# TODO: organize the system and instruction prompts into a separate module
SYS = """You are a technical documentation writer. You always write clear, concise, and accurate documentation for
 scientific experiments. Your documentation focuses on the experiment's purpose, procedure, and results. Therefore,
 details about specific python functions, packages, or libraries are not necessary. Your readers are experimental
 scientists.
"""

instr = """Please generate high-level two paragraph documentation for the following experiment. The first paragraph
 should explain the purpose and the second one the procedure, but don't use the word 'Paragraph'"""


@app.command()
def predict(data_file: str, model_path: str) -> None:
    run = mlflow.active_run()

    if run is None:
        run = mlflow.start_run()
    with run:
        logger.info(f"Active run_id: {run.info.run_id}")
        logger.info(f"running predict with {data_file}")
        logger.info(f"model path: {model_path}")

        # predictions = []
        with jsonlines.open(data_file) as reader:
            items = [item for item in reader]
            inputs = [item["instruction"] for item in items]
            labels = [item["output"] for item in items]

        pred = Predictor(model_path)
        timer_start = timer()
        predictions = pred.predict(SYS, instr, inputs)
        timer_end = timer()
        pred_time = timer_end - timer_start
        mlflow.log_metric("prediction_time/doc", pred_time / (len(inputs)))
        for i in range(len(inputs)):
            mlflow.log_text(labels[i], f"label_{i}.txt")
            mlflow.log_text(inputs[i], f"input_{i}.py")
            mlflow.log_text(predictions[i], f"prediction_{i}.txt")

        tokens = pred.tokenize(predictions)["input_ids"]
        total_tokens = sum([len(token) for token in tokens])
        mlflow.log_metric("total_tokens", total_tokens)
        mlflow.log_metric("tokens/sec", total_tokens / pred_time)


@app.command()
def import_model(model_name: str) -> None:
    pass


if __name__ == "__main__":
    logger.info(f"Torch version: {torch.__version__} , Cuda available: {torch.cuda.is_available()}")

    mlflow.autolog()
    app()