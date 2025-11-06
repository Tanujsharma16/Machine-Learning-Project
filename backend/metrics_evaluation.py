# metrics_evaluation.py
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)
from sklearn.model_selection import train_test_split

def evaluate_saved_models(X_test, y_test, model_paths):
    results = {}
    for name, path in model_paths.items():
        model = joblib.load(path)
        y_pred = model.predict(X_test)

        # probability handling for ROC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            if hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_test)
            else:
                y_prob = y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = float("nan")

        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "classification_report": classification_report(y_test, y_pred, digits=4)
        }

    return results


if __name__ == "__main__":
    print("🔍 Evaluating saved models...\n")

    # Load dataset
    data = pd.read_csv("diabetes.csv")
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    # Load scaler (the one used during training)
    scaler = joblib.load("models/lr_scaler.pkl")

    # Split dataset (same as training split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_test_scaled = scaler.transform(X_test)

    # Define saved model paths
    model_paths = {
        "GaussianNB": "models/gnb_model.pkl",
        "LogisticRegression": "models/log_reg_model.pkl"
    }

    # Evaluate and print results
    results = evaluate_saved_models(X_test_scaled, y_test, model_paths)

    for model_name, info in results.items():
        print(f"\n=== {model_name} ===")
        print(f"Accuracy: {info['accuracy']:.4f}")
        print(f"Precision: {info['precision']:.4f}")
        print(f"Recall: {info['recall']:.4f}")
        print(f"F1-score: {info['f1']:.4f}")
        print(f"ROC AUC: {info['roc_auc']:.4f}")
        print("Confusion Matrix:")
        print(info['confusion_matrix'])
        print("\nClassification Report:\n", info['classification_report'])
