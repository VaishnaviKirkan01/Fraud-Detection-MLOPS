
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import fraud_router
from backend.app.ml.model_service import FraudModelService


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading fraud detection model...")
    app.state.fraud_model = FraudModelService()
    logger.info("Fraud detection model loaded successfully.")

    yield

    app.state.fraud_model = None
    logger.info("Fraud detection API stopped.")


app = FastAPI(
    title="Fraud Detection MLOps API",
    description=(
        "Fraud detection using a trained Random Forest model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration for the existing React/Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Fraud Detection MLOps API is running",
        "docs": "/docs",
    }


app.include_router(fraud_router)