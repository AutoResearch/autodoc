from typing import List, Tuple

import nltk
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import single_meteor_score
from numpy import dot, mean, nan_to_num
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer


def eval_bleu_meteor(predictions: List[str], references: List[str]) -> Tuple[float, float]:
    nltk.download("wordnet")

    # Tokenize references
    tokenized_references = [ref.split() for ref in references]
    # Currently there is only 1 prediction for 1 reference, need to avg in future
    tokenized_predictions = [pred.split() if pred else [] for pred in predictions]

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
    meteor: float = nan_to_num(mean(meteor_scores), nan=0)

    return (bleu, meteor)


def eval_semscore(predictions: List[str], references: List[str]) -> float:
    """
    Calculate sentence embedding similarity score.
    https://arxiv.org/pdf/2401.17072.pdf
    """
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    def score(pred: str, ref: str) -> float:
        encodings = model.encode([pred, ref])
        assert len(encodings) == 2
        cos_dist: float = dot(encodings[0], encodings[1]) / norm(encodings[0]) * norm(encodings[1])
        return cos_dist

    scores = [score(pred, ref) for pred, ref in zip(predictions, references)]
    semscore: float = nan_to_num(mean(scores), nan=0)
    return semscore
