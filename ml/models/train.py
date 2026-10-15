
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "candidates"
REPORT_DIR = PROJECT_ROOT / "artifacts"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load the saved datasets
# --------------------------------------------------

def load_dataset(name):
    parquet_path = PROCESSED_DIR / f"{name}.parquet"
    csv_path = PROCESSED_DIR / f"{name}.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)

    raise FileNotFoundError(
        f"Dataset not found: {name}. "
        f"Checked {parquet_path} and {csv_path}"
    )


X_train = load_dataset("X_train_under")
y_train = load_dataset("y_train_under").squeeze()

X_val = load_dataset("X_val")
y_val = load_dataset("y_val").squeeze()

print("Datasets loaded successfully.")
print(f"Training shape: {X_train.shape}")
print(f"Validation shape: {X_val.shape}")

# Ensure consistent feature order
if list(X_train.columns) != list(X_val.columns):
    raise ValueError(
        "Training and validation feature columns do not match."
    )

if y_train.nunique() != 2 or y_val.nunique() != 2:
    raise ValueError("Expected both classes (0 and 1) in each dataset.")

if X_train.isnull().any().any() or X_val.isnull().any().any():
    raise ValueError("Missing feature values found. Handle them before training.")


# --------------------------------------------------
# 3. Define preprocessing
# --------------------------------------------------

categorical_features = X_train.select_dtypes(
    include=["object", "category", "str"]
).columns.tolist()

numerical_features = [
    col for col in X_train.columns
    if col not in categorical_features
]

print("\nCategorical features:", categorical_features)
print("Numerical features:", numerical_features)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        ("numerical", "passthrough", numerical_features),
    ],
    remainder="drop",
)


# --------------------------------------------------
# 4. Create the model pipeline
# --------------------------------------------------

classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight=None,
    random_state=42,
    n_jobs=-1,
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ]
)


# --------------------------------------------------
# 5. Train on balanced training data
# --------------------------------------------------

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

print("Training completed.")


# --------------------------------------------------
# 6. Evaluate on untouched validation data
# --------------------------------------------------

y_val_proba = model.predict_proba(X_val)[:, 1]

# Initial threshold for validation
threshold = 0.5
y_val_pred = (y_val_proba >= threshold).astype(int)

precision = precision_score(
    y_val, y_val_pred, zero_division=0
)
recall = recall_score(
    y_val, y_val_pred, zero_division=0
)
f1 = f1_score(
    y_val, y_val_pred, zero_division=0
)
pr_auc = average_precision_score(y_val, y_val_proba)
roc_auc = roc_auc_score(y_val, y_val_proba)
tn, fp, fn, tp = confusion_matrix(
    y_val, y_val_pred, labels=[0, 1]
).ravel()

print("\n--- Validation metrics at threshold 0.5 ---")
print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"F1-score:  {f1:.6f}")
print(f"PR-AUC:    {pr_auc:.6f}")
print(f"ROC-AUC:   {roc_auc:.6f}")
print(f"TN: {tn:,} | FP: {fp:,} | FN: {fn:,} | TP: {tp:,}")

print("\nClassification report:")
print(
    classification_report(
        y_val,
        y_val_pred,
        labels=[0, 1],
        target_names=["Legitimate", "Fraud"],
        zero_division=0,
    )
)


# --------------------------------------------------
# 7. Find a threshold that maximizes validation F1
# --------------------------------------------------

thresholds = np.arange(0.05, 0.96, 0.05)
threshold_results = []

for candidate_threshold in thresholds:
    candidate_pred = (
        y_val_proba >= candidate_threshold
    ).astype(int)

    candidate_precision = precision_score(
        y_val, candidate_pred, zero_division=0
    )
    candidate_recall = recall_score(
        y_val, candidate_pred, zero_division=0
    )
    candidate_f1 = f1_score(
        y_val, candidate_pred, zero_division=0
    )

    threshold_results.append(
        {
            "threshold": float(candidate_threshold),
            "precision": float(candidate_precision),
            "recall": float(candidate_recall),
            "f1": float(candidate_f1),
        }
    )

best_result = max(
    threshold_results,
    key=lambda result: result["f1"],
)

best_threshold = best_result["threshold"]

print("\n--- Best validation threshold by F1 ---")
print(f"Threshold: {best_threshold:.2f}")
print(f"Precision: {best_result['precision']:.6f}")
print(f"Recall:    {best_result['recall']:.6f}")
print(f"F1-score:  {best_result['f1']:.6f}")


# --------------------------------------------------
# 8. Save model and training metadata
# --------------------------------------------------

model_path = MODEL_DIR / "fraud_random_forest.joblib"

artifact = {
    "model": model,
    "threshold": best_threshold,
    "model_name": "Random Forest",
    "features": list(X_train.columns),
    "categorical_features": categorical_features,
    "numerical_features": numerical_features,
    "sklearn_version": sklearn.__version__,
}

joblib.dump(artifact, model_path)

report = {
    "model_name": "Random Forest",
    "threshold_selected_on_validation": best_threshold,
    "validation_metrics_at_threshold_0.5": {
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
    "best_validation_threshold_metrics": best_result,
    "threshold_search": threshold_results,
    "training_rows": int(len(X_train)),
    "validation_rows": int(len(X_val)),
    "features": list(X_train.columns),
    "sklearn_version": sklearn.__version__,
}

report_path = REPORT_DIR / "training_report.json"
report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("\nModel saved:", model_path)
print("Report saved:", report_path)
print("\nTraining pipeline finished successfully.")
