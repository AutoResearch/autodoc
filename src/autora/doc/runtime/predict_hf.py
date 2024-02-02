import logging
from typing import Dict, Iterable, List, Tuple

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from autora.doc.runtime.prompts import CODE_PLACEHOLDER, LLAMA2_INST_CLOSE

logger = logging.getLogger(__name__)

quantized_models = {"meta-llama/Llama-2-7b-chat-hf": "autora-doc/Llama-2-7b-chat-hf-nf4"}


def preprocess_code(code: str) -> str:
    lines: Iterable[str] = code.splitlines()
    skip_starts = {"import", "from", "#"}
    lines = filter(
        lambda line: not (any([line.strip().startswith(skip) for skip in skip_starts]) or line.strip() == ""),
        lines,
    )
    return "\n".join(lines)


class Predictor:
    def __init__(self, input_model_path: str):
        model_path, config = Predictor.get_config(input_model_path)
        if model_path != input_model_path:
            logger.info(f"Mapped requested model '{input_model_path}' to '{model_path}'")

        logger.info(f"Loading model from {model_path} using config {config}")
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

    @staticmethod
    def get_config(model_path: str) -> Tuple[str, Dict[str, str]]:
        if torch.cuda.is_available():
            from transformers import BitsAndBytesConfig

            config = {"device_map": "auto"}
            mapped_path = quantized_models.get(model_path, None)
            if mapped_path:
                # found an already quantized model, so no need to get a new quant config
                return mapped_path, config

            # Load the model in 4bit quantization for faster inference on smaller GPUs
            config["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            return model_path, config
        else:
            return model_path, {}
