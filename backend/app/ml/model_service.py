
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "candidates"
    / "fraud_random_forest.joblib"
)


class FraudModelService:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Fraud model not found: {MODEL_PATH}"
            )

        artifact = joblib.load(MODEL_PATH)

        self.model = artifact["model"]
        self.threshold = float(artifact["threshold"])
        self.model_name = artifact["model_name"]
        self.features = artifact["features"]

    def predict(self, transaction: dict) -> dict:
        row = pd.DataFrame([transaction])
        row = row[self.features]

        probability = float(
            self.model.predict_proba(row)[0, 1]
        )

        is_fraud = probability >= self.threshold

        return {
            "prediction": (
                "FRAUD" if is_fraud else "LEGITIMATE"
            ),
            "is_fraud": is_fraud,
            "fraud_probability": probability,
            "threshold": self.threshold,
            "model_name": self.model_name,
            "message": (
                "Transaction flagged for fraud review."
                if is_fraud
                else "Transaction classified as legitimate."
            ),
        }
