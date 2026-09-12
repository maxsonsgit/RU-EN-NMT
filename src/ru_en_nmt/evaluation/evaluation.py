from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def bleu(references, hypotheses):
    bleu = corpus_bleu(
        references, 
        hypotheses,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=SmoothingFunction().method4
        )

    return bleu * 100
