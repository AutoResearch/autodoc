from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EvalResult:
    """Class for storing LLM evaluation results"""

    prediction: List[str]
    prompt: str
    bleu_score: Optional[float] = None
    meteor_score: Optional[float] = None
