import pandas as pd
import re
import os

import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ========================
# PREPROCESS
# ========================
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-ząćęłńóśźż0-9 ]', '', text)
    return text

# ========================
# LOAD DATA
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "spam.csv")

df = pd.read_csv(csv_path)
df["text"] = df["text"].apply(preprocess)

# ========================
# TRAIN MODEL
# ========================
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(df["text"])
y = df["label"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Zapisano model i vectorizer")