from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-ząćęłńóśźż0-9 ]', '', text)
    return text

app = FastAPI()

class Request(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: list[str]

@app.get("/")
def home():
    return {"status": "API działa"}

@app.post("/predict")
def predict(req: Request):
    text = preprocess(req.text)
    X_new = vectorizer.transform([text])
    pred = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0].max()

    return {
        "text": req.text,
        "prediction": pred,
        "confidence": round(float(proba), 3)
    }

@app.post("/predict_batch")
def predict_batch(req: BatchRequest):
    processed = [preprocess(t) for t in req.texts]
    X = vectorizer.transform(processed)
    preds = model.predict(X)
    probas = model.predict_proba(X).max(axis=1)
    return [
        {
            "text": req.texts[i],
            "prediction": preds[i],
            "confidence": float(probas[i])
        }
        for i in range(len(req.texts))
    ]