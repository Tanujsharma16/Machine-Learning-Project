# ===============================
# Diabetes Prediction Model Trainer
# Naive Bayes + Logistic Regression
# ===============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os


# -------------------------------
# 1. Load dataset
# -------------------------------
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


# -------------------------------
# 2. Clean data (replace zeros)
# -------------------------------
def clean_data(data):
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_cols:
        non_zero_mean = data.loc[data[col] > 0, col].mean()
        data.loc[data[col] == 0, col] = non_zero_mean
    return data


# -------------------------------
# 3. Preprocess data (split + scale)
# -------------------------------
def preprocess_data(data, target_column='Outcome'):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# -------------------------------
# 4. Train models
# -------------------------------
def train_models(X_train, y_train):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)

    return gnb, log_reg


# -------------------------------
# 5. Evaluate models
# -------------------------------
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n=== Evaluation Report for {model_name} ===")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.show()


# -------------------------------
# 6. Save model and scaler
# -------------------------------
def save_model(model, scaler, model_path, scaler_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"✅ Saved model to: {model_path}")
    print(f"✅ Saved scaler to: {scaler_path}")


# -------------------------------
# 7. Exploratory Data Analysis (optional)
# -------------------------------
def perform_eda(data):
    print("\nData Summary:")
    print(data.describe())

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    # Distribution plots
    data.hist(figsize=(12, 10))
    plt.suptitle("Feature Distributions")
    plt.show()


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    # Step 1: Load
    data = load_data('diabetes.csv')

    # Step 2: Clean
    data = clean_data(data)

    # Step 3: EDA (optional visualization)
    perform_eda(data)

    # Step 4: Preprocess
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data, target_column='Outcome')

    # Step 5: Train
    gnb_model, log_reg_model = train_models(X_train, y_train)

    # Step 6: Evaluate
    evaluate_model(gnb_model, X_test, y_test, model_name='Gaussian Naive Bayes')
    evaluate_model(log_reg_model, X_test, y_test, model_name='Logistic Regression')

    # Step 7: Save
    save_model(gnb_model, scaler, 'models/gnb_model.pkl', 'models/gnb_scaler.pkl')
    save_model(log_reg_model, scaler, 'models/log_reg_model.pkl', 'models/lr_scaler.pkl')
