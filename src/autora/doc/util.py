import json
from typing import Any, Dict, List, Tuple

from autora.doc.runtime.prompts import PromptBuilder


def load_file(json_file_path: str) -> List[Dict[str, Any]]:
    # Read and parse the JSON file
    with open(json_file_path, "r") as file:
        data: List[Dict[str, Any]] = json.load(file)
    return data


def get_prompts_from_file(prompts_file: str) -> List[str]:
    prompts_data = load_file(prompts_file)
    prompts_list = [PromptBuilder(p["SYS"], p["INSTR"]).build() for p in prompts_data]
    return prompts_list
