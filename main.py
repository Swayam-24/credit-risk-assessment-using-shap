import logging
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "credit_risk_model.pkl"
THRESHOLD_PATH = BASE_DIR / "best_threshold.pkl"
STATIC_DIR = BASE_DIR / "static"

FEATURE_COLUMNS = (
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model = joblib.load(MODEL_PATH)
        threshold = float(joblib.load(THRESHOLD_PATH))

        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("The saved classification threshold must be between 0 and 1.")
        if not callable(getattr(model, "predict_proba", None)):
            raise TypeError("The saved model does not provide predict_proba.")

        model_classes = getattr(model, "classes_", None)
        if model_classes is not None and tuple(model_classes) != (0, 1):
            raise ValueError("The saved model must expose binary classes 0 and 1 in that order.")

        model_features = getattr(model, "feature_names_in_", None)
        if model_features is not None and tuple(model_features) != FEATURE_COLUMNS:
            raise ValueError("The saved model feature names do not match the API input schema.")

        app.state.model = model
        app.state.threshold = threshold
        logger.info("Model artifacts loaded successfully.")
    except Exception:
        logger.exception("Failed to load the model artifacts.")
        raise

    yield


app = FastAPI(
    title="Credit Risk Assessment",
    description="Credit-risk estimates from the trained model.",
    lifespan=lifespan,
)

cors_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]
if "*" in cors_origins:
    raise ValueError("CORS_ORIGINS must list explicit trusted origins, not '*'.")
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )


class LoanApplication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    person_age: int = Field(ge=18, le=100)
    person_income: float = Field(gt=0, allow_inf_nan=False)
    person_home_ownership: Literal["RENT", "MORTGAGE", "OWN", "OTHER"]
    person_emp_length: float = Field(ge=0, le=60, allow_inf_nan=False)
    loan_intent: Literal[
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION",
    ]
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    loan_amnt: float = Field(gt=0, allow_inf_nan=False)
    loan_int_rate: float = Field(ge=0, le=40, allow_inf_nan=False)
    loan_percent_income: float = Field(ge=0, le=1, allow_inf_nan=False)
    cb_person_default_on_file: Literal["N", "Y"]
    cb_person_cred_hist_length: int = Field(ge=0, le=60)


class PredictionResponse(BaseModel):
    default_probability: float = Field(ge=0, le=1)
    default_prediction: Literal[0, 1]
    threshold: float = Field(ge=0, le=1)
    Result: Literal["High Risk", "Low Risk"]


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    if not hasattr(request.app.state, "model"):
        raise HTTPException(status_code=503, detail="Model is not ready.")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: LoanApplication, request: Request) -> PredictionResponse:
    model = request.app.state.model
    threshold = request.app.state.threshold
    input_df = pd.DataFrame(
        [data.model_dump()],
        columns=FEATURE_COLUMNS,
    )

    try:
        probabilities = np.asarray(model.predict_proba(input_df), dtype=float)
        if probabilities.shape != (1, 2):
            raise ValueError("The model returned an unexpected probability shape.")

        probability = float(probabilities[0, 1])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("The model returned an invalid probability.")
    except Exception as exc:
        logger.exception("Prediction failed.")
        raise HTTPException(
            status_code=500,
            detail="Prediction could not be completed. Please try again later.",
        ) from exc

    prediction: Literal[0, 1] = 1 if probability >= threshold else 0
    return PredictionResponse(
        default_probability=probability,
        default_prediction=prediction,
        threshold=threshold,
        Result="High Risk" if prediction else "Low Risk",
    )


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )
