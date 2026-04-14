import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import re

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-ząćęłńóśźż0-9 ]', '', text)
    return text

def main():
    df = pd.read_csv("spam.csv")

    print("=== HEAD ===")
    print(df.head())

    print("\n=== ROZKŁAD KLAS ===")
    print(df["label"].value_counts())

    vectorizer = TfidfVectorizer(ngram_range=(1,2))

    df["text"] = df["text"].apply(preprocess)
    X = vectorizer.fit_transform(df["text"])
    y = df["label"]

    print("\n=== KSZTAŁT DANYCH ===")
    print(X.shape)

    print("\n=== SŁOWNIK ===")
    print(vectorizer.get_feature_names_out())

    print("\n=== PRZYKŁADOWY WEKTOR ===")
    print(X[0].toarray())

    #podział danych
    X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
        X, y, df["text"], test_size=0.25, random_state=42
    )

    #model
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    #predykcja
    y_pred = model.predict(X_test)

    print("\n=== WYNIK ===")
    
    for text, real, pred in zip(text_test, y_test, y_pred):
        print(f"{text} | real: {real} | pred: {pred}")

    nowa = ["Wygrałeś 1000 zł kliknij tutaj"]

    X_new = vectorizer.transform(nowa)
    pred = model.predict(X_new)

    print("\n=== NOWA WIADOMOŚĆ ===")
    print(nowa[0])
    print("Predykcja: ", pred[0])

    accuracy = accuracy_score(y_test, y_pred)

    #1.0   → 100% (wszystko trafione)
    #0.5   → 50% (losowo)
    #0.75  → 75% (już coś działa)
    print("\n=== ACCURACY ===")
    print(accuracy)

    print("\n=== RAPORT ===")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()