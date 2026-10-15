
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

# Project paths
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

    raise FileNotFoundError(
        f"Dataset not found: {name} in {PROCESSED_DIR}"
    )


# Load the existing model and held-out test data
X_test = load_dataset("X_test")
y_test = load_dataset("y_test").squeeze()

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
expected_features = artifact["features"]
current_threshold = float(artifact["threshold"])

if list(X_test.columns) != expected_features:
    raise ValueError(
        "Test features do not match the trained model."
    )

if len(X_test) != len(y_test):
    raise ValueError(
        "X_test and y_test have different row counts."
    )

print("Model:", artifact["model_name"])
print("Existing threshold:", current_threshold)
print("Test rows:", f"{len(X_test):,}")

# Predict probabilities once; no retraining
y_true = y_test.astype(int).to_numpy()
fraud_probabilities = model.predict_proba(X_test)[:, 1]

print(
    "PR-AUC:",
    f"{average_precision_score(y_true, fraud_probabilities):.6f}",
)
print(
    "ROC-AUC:",
    f"{roc_auc_score(y_true, fraud_probabilities):.6f}",
)

# Compare decision thresholds
thresholds = [0.80, 0.90, 0.95]
results = []

for threshold in thresholds:
    predictions = (
        fraud_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_true, predictions, zero_division=0
    )
    recall = recall_score(
        y_true, predictions, zero_division=0
    )
    f1 = f1_score(
        y_true, predictions, zero_division=0
    )

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "predicted_fraud": int(tp + fp),
        "actual_fraud": int(tp + fn),
    })

comparison = pd.DataFrame(results)

print("\n--- THRESHOLD COMPARISON ---")
print(
    comparison.to_string(
        index=False,
        formatters={
            "threshold": "{:.2f}".format,
            "precision": "{:.4f}".format,
            "recall": "{:.4f}".format,
            "f1_score": "{:.4f}".format,
        },
    )
)

report_path = REPORT_DIR / "threshold_comparison.csv"
comparison.to_csv(report_path, index=False)

print("\nComparison saved to:", report_path)
print("Model was not retrained or modified.")
