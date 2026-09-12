import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


names = ["number_of_russian_sentense", "russian_sentense", "number_of_english_sentense", "english_sentense"]
data = pd.read_csv(f"{Path(__file__).resolve().parents[3]}/data/raw/Tatoeba Sentence pairs in Russian-English - 2026-06-27.tsv", sep="\t", on_bad_lines="skip", names=names)

N = 30
data = data[data["english_sentense"].str.split().str.len() <= N]
X = data[["number_of_russian_sentense", "russian_sentense"]]
y = data[["number_of_english_sentense", "english_sentense"]]

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=11)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=11)

train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)
val = pd.concat([X_val, y_val], axis=1)

train.to_csv(f"{Path(__file__).resolve().parents[3]}/data/preprocesed/train.csv", index=False)
test.to_csv(f"{Path(__file__).resolve().parents[3]}/data/preprocesed/test.csv", index=False)
val.to_csv(f"{Path(__file__).resolve().parents[3]}/data/preprocesed/val.csv", index=False)