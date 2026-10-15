
from functools import lru_cache
from pathlib import Path
import logging

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from backend.app.schemas.fraud import (
    PredictionResponse,
    TransactionInput,
)
from backend.app.services.fraud_service import (
    FraudPredictionService,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Fraud Detection"])

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "paysim.csv"


@router.get("/health")
def health(request: Request):
    model_service = getattr(
        request.app.state, "fraud_model", None
    )

    return {
        "status": "healthy",
        "model_loaded": model_service is not None,
        "model_name": (
            model_service.model_name
            if model_service is not None
            else None
        ),
        "threshold": (
            model_service.threshold
            if model_service is not None
            else None
        ),
    }


@router.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    transaction: TransactionInput,
    request: Request,
):
    model_service = getattr(
        request.app.state, "fraud_model", None
    )

    if model_service is None:
        raise HTTPException(
            status_code=503,
            detail="Fraud detection model is not loaded.",
        )

    service = FraudPredictionService(model_service)

    try:
        return service.predict(transaction)
    except Exception as exc:
        logger.exception("Fraud prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc


@lru_cache(maxsize=1)
def load_dataset_analytics():
    """Calculate dataset statistics once and cache the result."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    columns = [
        "step",
        "type",
        "amount",
        "isFraud",
    ]

    total = 0
    fraud_total = 0
    total_amount = 0.0
    fraud_amount = 0.0

    type_counts = {}
    step_stats = {}
    sample_frames = []

    # Process the large dataset in chunks to limit memory use.
    for chunk in pd.read_csv(
        DATASET_PATH,
        usecols=columns,
        chunksize=200_000,
    ):
        chunk["isFraud"] = chunk["isFraud"].astype(int)

        total += len(chunk)
        fraud_total += int(chunk["isFraud"].sum())
        total_amount += float(chunk["amount"].sum())

        fraud_rows = chunk[chunk["isFraud"] == 1]
        fraud_amount += float(fraud_rows["amount"].sum())

        for transaction_type, count in (
            chunk["type"].value_counts().items()
        ):
            type_counts[transaction_type] = (
                type_counts.get(transaction_type, 0)
                + int(count)
            )

        grouped = chunk.groupby("step").agg(
            transactions=("isFraud", "size"),
            fraud_transactions=("isFraud", "sum"),
            amount=("amount", "sum"),
        )

        for step, row in grouped.iterrows():
            key = int(step)
            current = step_stats.setdefault(
                key,
                {
                    "step": key,
                    "transactions": 0,
                    "fraud_transactions": 0,
                    "amount": 0.0,
                },
            )
            current["transactions"] += int(row["transactions"])
            current["fraud_transactions"] += int(
                row["fraud_transactions"]
            )
            current["amount"] += float(row["amount"])

        # Collect a small sample for the dashboard table.
        sample_frames.append(
            chunk.sample(
                n=min(5, len(chunk)),
                random_state=42,
            )
        )

    if total == 0:
        raise ValueError("The dataset is empty.")

    samples = pd.concat(
        sample_frames,
        ignore_index=True,
    ).sample(
        n=min(20, sum(len(frame) for frame in sample_frames)),
        random_state=42,
    )

    samples = samples.rename(
        columns={"isFraud": "dataset_label"}
    )

    samples["dataset_label"] = samples[
        "dataset_label"
    ].map({1: "FRAUD", 0: "LEGITIMATE"})

    return {
        "dataset_source": "data/raw/paysim.csv",
        "total_transactions": total,
        "fraud_transactions": fraud_total,
        "legitimate_transactions": total - fraud_total,
        "fraud_rate_percent": round(
            fraud_total / total * 100, 5
        ),
        "total_amount": round(total_amount, 2),
        "fraud_amount": round(fraud_amount, 2),
        "transaction_types": [
            {"type": name, "count": count}
            for name, count in sorted(type_counts.items())
        ],
        "fraud_by_step": [
            {
                **item,
                "amount": round(item["amount"], 2),
            }
            for _, item in sorted(step_stats.items())
        ],
        "sample_transactions": samples.to_dict(
            orient="records"
        ),
    }


@router.get("/analytics/overview")
def analytics_overview():
    try:
        return load_dataset_analytics()
    except FileNotFoundError as exc:
        logger.exception("PaySim dataset not found")
        raise HTTPException(
            status_code=404,
            detail="PaySim dataset file was not found.",
        ) from exc
    except Exception as exc:
        logger.exception("Dataset analytics failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to calculate dataset analytics.",
        ) from exc
