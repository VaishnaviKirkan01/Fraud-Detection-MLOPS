
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TransactionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=0)
    type: Literal[
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER",
    ]
    amount: float = Field(ge=0, allow_inf_nan=False)
    oldbalanceOrg: float = Field(ge=0, allow_inf_nan=False)
    newbalanceOrig: float = Field(ge=0, allow_inf_nan=False)
    oldbalanceDest: float = Field(ge=0, allow_inf_nan=False)
    newbalanceDest: float = Field(ge=0, allow_inf_nan=False)


class PredictionResponse(BaseModel):
    prediction: Literal["FRAUD", "LEGITIMATE"]
    is_fraud: bool
    fraud_probability: float
    threshold: float
    model_name: str
    message: str
