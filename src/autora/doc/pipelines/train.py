from typing import Dict, Iterable

import torch
from datasets import Dataset
from numba import cuda
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

from autora.doc.pipelines.data import load_data, preprocess_code
from autora.doc.runtime.prompts import INSTR_SWEETP_1, SYS_GUIDES, PromptBuilder


def get_dataset(jsonl_path: str) -> Dataset:
    # "instruction", "output"
    inputs, labels = load_data(jsonl_path)

    def gen() -> Iterable[Dict[str, str]]:
        for i, o in zip(inputs, labels):
            text = PromptBuilder(SYS_GUIDES, INSTR_SWEETP_1).add_example(preprocess_code(i), o).build()
            yield {"text": text}

    ds = Dataset.from_generator(gen)
    return ds


def fine_tune(base_model: str, new_model_name: str, dataset: Dataset) -> None:
    compute_dtype = getattr(torch, "float16")
    device = cuda.get_current_device()
    device.reset()

    # train using 4 bit quantization for lower GPU memory usage
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=quant_config, device_map={"": 0}
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # All of these parameters are initial defaults and may need further tuning
    training_params = TrainingArguments(
        output_dir="./results",
        num_train_epochs=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=1,  # TODO: Increase once there's more data
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard",
    )

    # Use a Supervised Fine-Tuning Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_params,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )

    trainer.train()
    trainer.model.save_pretrained(new_model_name)
    trainer.tokenizer.save_pretrained(new_model_name)
