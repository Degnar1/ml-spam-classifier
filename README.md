# ML Spam Classifier API

This project is a simple machine learning API for spam detection.  
It uses TF-IDF vectorization and Logistic Regression to classify text messages as spam or ham.

## Features

- Single text prediction (`/predict`)
- Batch prediction (`/predict_batch`)
- Confidence score for each prediction
- Pre-trained model loaded from disk (no training during API startup)

## Tech Stack

- Python
- FastAPI
- scikit-learn
- pandas

## How it works

1. Text is preprocessed (lowercase, cleaned)
2. Transformed into numerical features using TF-IDF
3. Classified using Logistic Regression
4. API returns prediction and confidence score

## Run locally

Install dependencies:

pip install -r requirements.txt

Train the model:

python train.py

Run the API:

uvicorn api:app --reload

## Example request

### POST /predict

{
  "text": "Wygrałeś iPhone kliknij teraz"
}

## Example response

{
  "text": "Wygrałeś iPhone kliknij teraz",
  "prediction": "spam",
  "confidence": 0.698
}

## Batch example

### POST /predict_batch

{
  "texts": [
    "Wygrałeś iPhone kliknij teraz",
    "Spotkamy się jutro"
  ]
}

## Notes

This project demonstrates:

- building an API for machine learning models
- separating training and inference
- handling both single and batch predictions