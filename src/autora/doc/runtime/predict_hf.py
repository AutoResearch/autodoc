import logging
from typing import Dict, Iterable, List

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from autora.doc.runtime.prompts import CODE_PLACEHOLDER, LLAMA2_INST_CLOSE

logger = logging.getLogger(__name__)


def preprocess_code(code: str) -> str:
    lines: Iterable[str] = code.splitlines()
    skip_starts = {"import", "from", "#"}
    lines = filter(
        lambda line: not (any([line.strip().startswith(skip) for skip in skip_starts]) or line.strip() == ""),
        lines,
    )
    return "\n".join(lines)


class Predictor:
    def __init__(self, model_path: str):
        config = self.get_config()

        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **config,
        )
        logger.info("Model loaded")
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def predict(
        self,
        prompt_template: str,
        inputs: List[str],
        do_sample: float = 0.0,
        temperature: float = 0.01,
        top_p: float = 0.95,
        top_k: float = 1,
        max_new_tokens: float = 2048,
        num_ret_seq: float = 1,
    ) -> List[str]:
        # convert to bool in case it came in as a generate float param from the CLI
        do_sample = bool(do_sample)
        logger.info(
            f"Generating {len(inputs)} predictions. do_sample: {do_sample}, temperature: {temperature}, top_p: {top_p},"
            f" top_k: {top_k}, max_new_tokens: {max_new_tokens}"
        )
        prompts = [
            prompt_template.replace(CODE_PLACEHOLDER, preprocess_code(input).strip("\n")) for input in inputs
        ]
        sequences = self.pipeline(
            prompts,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=int(top_k),
            num_return_sequences=int(num_ret_seq),
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=int(max_new_tokens),
        )

        results = [Predictor.trim_prompt(seq["generated_text"]) for sequence in sequences for seq in sequence]
        logger.info(f"Generated {len(results)} results")
        return results

    @staticmethod
    def trim_prompt(output: str) -> str:
        marker = output.rfind(LLAMA2_INST_CLOSE)
        if marker == -1:
            logger.warning(f"Could not find end of prompt marker '{LLAMA2_INST_CLOSE}' in '{output}'")
            return output
        return output[marker + len(LLAMA2_INST_CLOSE) :]

    def tokenize(self, input: List[str]) -> Dict[str, List[List[int]]]:
        tokens: Dict[str, List[List[int]]] = self.tokenizer(input)
        return tokens

    def get_config(self) -> Dict[str, str]:
        if torch.cuda.is_available():
            from transformers import BitsAndBytesConfig

            # Load the model in 4bit quantization for faster inference on smaller GPUs
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ),
                "device_map": "auto",
            }
        else:
            return {}
