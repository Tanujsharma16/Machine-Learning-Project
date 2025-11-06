from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np


# ==========================
# Load saved models and scaler
# ==========================
# Make sure these file names match what you saved in train_model.py
nb = joblib.load('models/gnb_model.pkl')
lr = joblib.load('models/log_reg_model.pkl')
scaler = joblib.load('models/lr_scaler.pkl')


# Initialize FastAPI app
app = FastAPI(title="Diabetes Prediction API")


# ==========================
# Input Schema
# ==========================
class Patient(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


# ==========================
# Prediction Route
# ==========================
@app.post("/predict")
def predict(patient: Patient, model: str = "lr"):
    # Convert input into numpy array in same order as training
    features = np.array([[
        patient.Pregnancies,
        patient.Glucose,
        patient.BloodPressure,
        patient.SkinThickness,
        patient.Insulin,
        patient.BMI,
        patient.DiabetesPedigreeFunction,
        patient.Age
    ]])

    # Scale input
    features_scaled = scaler.transform(features)

    # Select model (lr or nb)
    chosen_model = lr if model == "lr" else nb

    # Prediction
    if hasattr(chosen_model, "predict_proba"):
        prob = float(chosen_model.predict_proba(features_scaled)[0, 1])
    else:
        prob = float(chosen_model.predict(features_scaled)[0])

    pred = int(chosen_model.predict(features_scaled)[0])
    label = "✅ Non-Diabetic" if pred == 0 else "⚠️ Diabetic - High Risk"

    return {
        "prediction": label,
        "probability": round(prob, 4),
        "model_used": "Logistic Regression" if model == "lr" else "Naive Bayes"
    }


# ==========================
# Root Endpoint
# ==========================
@app.get("/")
def root():
    return {"message": "Diabetes Prediction API is running!"}

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # sab allowed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
