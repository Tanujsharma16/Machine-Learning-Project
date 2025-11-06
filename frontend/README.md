# 🧠 Diabetes Prediction using Naïve Bayes + Logistic Regression (MERN + FastAPI)

## 📌 Objective
Predict the likelihood of diabetes in patients using ML models integrated with a MERN-style frontend.

## 🚀 Features
- Data Cleaning, EDA, and Model Training
- Naïve Bayes & Logistic Regression models
- REST API built using FastAPI
- React Frontend Form for input
- Real-time prediction with probability

## 🧩 Tech Stack
**Backend:** FastAPI, Scikit-learn, Pandas, Joblib  
**Frontend:** ReactJS  
**Dataset:** PIMA Indians Diabetes Dataset (Kaggle)

## ⚙️ How to Run

### Backend
```bash
cd backend
pip install -r requirements.txt
python train_model.py
uvicorn main:app --reload
