from typing import List, Tuple

import nltk
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import single_meteor_score


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
    meteor = sum(meteor_scores) / len(predictions) if predictions else 0

    return (bleu, meteor)
