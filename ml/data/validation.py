from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

NUMERIC_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


def validate_raw_dataframe(df: pd.DataFrame) -> None:
    """Validate the structure and key values of a PaySim DataFrame.

    Raises:
        ValueError: If the dataframe is empty or has invalid structure/values.
    """
    if df.empty:
        raise ValueError("The dataset is empty.")

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df[REQUIRED_COLUMNS].isna().any().any():
        missing_counts = df[REQUIRED_COLUMNS].isna().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(f"Missing values found in required columns: {missing_counts}")

    for column in NUMERIC_COLUMNS:
        converted = pd.to_numeric(df[column], errors="coerce")
        invalid = converted.isna()
        if invalid.any():
            raise ValueError(
                f"Column '{column}' contains {int(invalid.sum())} non-numeric value(s)."
            )

    fraud_values = set(pd.to_numeric(df["isFraud"], errors="raise").unique())
    if not fraud_values.issubset({0, 1}):
        raise ValueError(f"'isFraud' must contain only 0 and 1; found {fraud_values}.")

    flagged_values = set(pd.to_numeric(df["isFlaggedFraud"], errors="raise").unique())
    if not flagged_values.issubset({0, 1}):
        raise ValueError(
            "'isFlaggedFraud' must contain only 0 and 1; "
            f"found {flagged_values}."
        )

    if (pd.to_numeric(df["amount"], errors="raise") < 0).any():
        raise ValueError("'amount' contains negative values.")

    for column in [
        "step", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest"
    ]:
        if (pd.to_numeric(df[column], errors="raise") < 0).any():
            raise ValueError(f"'{column}' contains negative values.")

    print("Raw dataset validation passed.")
