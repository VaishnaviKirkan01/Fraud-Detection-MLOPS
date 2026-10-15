
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "candidates"
    / "fraud_random_forest.joblib"
)
REPORT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(name):
    parquet_path = PROCESSED_DIR / f"{name}.parquet"
    csv_path = PROCESSED_DIR / f"{name}.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)

    raise FileNotFoundError(f"Dataset not found: {name}")


# Load test data and the saved model artifact
X_test = load_dataset("X_test")
y_test = load_dataset("y_test").squeeze()

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
threshold = float(artifact["threshold"])
expected_features = artifact["features"]

if list(X_test.columns) != expected_features:
    raise ValueError("Test features do not match the trained model.")

print("Loaded model:", artifact["model_name"])
print("Selected validation threshold:", threshold)
print("Test shape:", X_test.shape)

# Evaluate the untouched test set
y_test_proba = model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= threshold).astype(int)

precision = precision_score(
    y_test, y_test_pred, zero_division=0
)
recall = recall_score(
    y_test, y_test_pred, zero_division=0
)
f1 = f1_score(
    y_test, y_test_pred, zero_division=0
)
pr_auc = average_precision_score(y_test, y_test_proba)
roc_auc = roc_auc_score(y_test, y_test_proba)

tn, fp, fn, tp = confusion_matrix(
    y_test, y_test_pred, labels=[0, 1]
).ravel()

print("\n--- FINAL TEST RESULTS ---")
print(f"Threshold: {threshold:.4f}")
print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"F1-score:  {f1:.6f}")
print(f"PR-AUC:    {pr_auc:.6f}")
print(f"ROC-AUC:   {roc_auc:.6f}")
print(f"TN: {tn:,} | FP: {fp:,} | FN: {fn:,} | TP: {tp:,}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_test_pred,
        labels=[0, 1],
        target_names=["Legitimate", "Fraud"],
        zero_division=0,
    )
)

report = {
    "model_name": artifact["model_name"],
    "threshold": threshold,
    "test_metrics": {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    },
    "test_rows": int(len(X_test)),
}

report_path = REPORT_DIR / "test_evaluation_report.json"
report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("\nTest report saved:", report_path)
print("Final test evaluation completed.")
