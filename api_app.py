# api_app.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Bank Product Adoption Predictor API")

# Load trained model
model = joblib.load("best_model.pkl")

# Define input schema
class ClientData(BaseModel):
    age: int
    job: int
    marital: int
    education: int
    default: int
    balance: float
    housing: int
    loan: int
    contact: int
    day: int
    month: int
    duration: float
    campaign: int
    pdays: int
    previous: int
    poutcome: int
    

@app.get("/")
def root():
    return {"message": "Bank Product Adoption Predictor API is live!"}

@app.post("/predict")
def predict(data: ClientData):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]
    result = "Yes" if prediction == 1 else "No"
    return {
        "prediction": result,
        "probability_of_yes": round(proba, 3)
    }
