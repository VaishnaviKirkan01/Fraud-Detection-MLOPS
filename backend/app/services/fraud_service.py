
from backend.app.ml.model_service import FraudModelService
from backend.app.schemas.fraud import (
    PredictionResponse,
    TransactionInput,
)


class FraudPredictionService:
    def __init__(self, model_service: FraudModelService):
        self.model_service = model_service

    def predict(
        self,
        transaction: TransactionInput,
    ) -> PredictionResponse:
        result = self.model_service.predict(
            transaction.model_dump()
        )

        return PredictionResponse(**result)
