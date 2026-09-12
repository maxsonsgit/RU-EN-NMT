from nltk.translate import AlignedSent
from nltk.translate.ibm1 import IBMModel1
from sacremoses import MosesTokenizer, MosesDetokenizer
import pandas as pd
import numpy as np
from pathlib import Path

from evaluation import bleu


def tokenize(text: str, lang: str):
    if lang == "ru":
        tokens = tokenizer_ru.tokenize(text)
    else:
        tokens = tokenizer_en.tokenize(text)

    tokens = [token.lower() for token in tokens]
    return tokens

def translate(text_tok, model):
    translation = []

    for word in text_tok:
        if word in model.translation_table:
            best_transtaltion = max(model.translation_table[word], key=model.translation_table[word].get)
            translation.append(best_transtaltion)
        else:
            translation.append(word)

    return translation

train = pd.read_csv(f"{Path(__file__).resolve().parents[3]}/data/preprocesed/train.csv")
test = pd.read_csv(f"{Path(__file__).resolve().parents[3]}/data/preprocesed/test.csv")
val = pd.read_csv(f"{Path(__file__).resolve().parents[3]}/data/preprocesed/val.csv")

X_train, y_train = train["russian_sentense"], train["english_sentense"]
X_test, y_test = test["russian_sentense"], test["english_sentense"]
X_val, y_val = val["russian_sentense"], val["english_sentense"]

tokenizer_ru = MosesTokenizer(lang="ru")
tokenizer_en = MosesTokenizer(lang="en")

print("tokenizing")
X_train_token = [tokenize(el, lang="ru") for el in X_train]
y_train_token = [tokenize(el, lang="en") for el in y_train]

X_test_token = [tokenize(el, lang="ru") for el in X_test]
y_test_token = [tokenize(el, lang="en") for el in y_test]

print("aligning")
train_bitext = [AlignedSent(X_train_token[i], y_train_token[i]) for i in range(len(X_train_token))]
test_bitext = [AlignedSent(X_test_token[i], y_test_token[i]) for i in range(len(X_test_token))]

print("learning")
model = IBMModel1(train_bitext, 5)

print("evaluating")
detoketizer = MosesDetokenizer(lang="en")
translotion_sample = translate(X_test_token[0], model)
en_sentense = detoketizer.detokenize(translotion_sample)

hypotheses = [translate(sentence, model) for sentence in X_test_token]
references = [[ref] for ref in y_test_token]

print()
print("-" * 100)
print(f"{bleu(references, hypotheses):.4f}")
print("-" * 100)