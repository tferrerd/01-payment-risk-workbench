"""Local FastAPI service for a validated development candidate."""
from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, create_model

from .pipeline import DEFAULT_ARTIFACT, FEATURES, PaymentRiskPipeline
from .monitoring import MonitoringMiddleware, RELEASE_ID, Telemetry, source_revision

Transaction = create_model(
    "Transaction",
    __config__=ConfigDict(extra="forbid", strict=True, allow_inf_nan=False),
    **{feature: (float, Field(..., ge=0) if feature in ["Time", "Amount"] else Field(...)) for feature in FEATURES},
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)


class PredictionResult(BaseModel):
    fraud_probability: float = Field(ge=0, le=1, allow_inf_nan=False)
    decision: Literal["approve", "cancel"]
    modeled_approval_profit: float = Field(allow_inf_nan=False)
    modeled_policy_profit: float = Field(allow_inf_nan=False)
    model_id: str
    scenario: str
    model_status: Literal["development_candidate"]
    probability_calibrated: Literal[False]


class PredictionResponse(BaseModel):
    model_id: str
    model_status: Literal["development_candidate"]
    scenario: str
    threshold: float
    predictions: list[PredictionResult]


@asynccontextmanager
async def lifespan(app):
    app.state.pipeline = PaymentRiskPipeline.load(Path(os.environ.get("PAYMENT_RISK_MODEL_PATH", str(DEFAULT_ARTIFACT))))
    app.state.telemetry = Telemetry()
    yield
    del app.state.pipeline


app = FastAPI(title="Payment Risk Workbench", version="0.1.0",
              description="Development-candidate inference under assumed 0.3% legitimate margin and 100% fraud loss. Scores have not been post-hoc calibrated.",
              lifespan=lifespan)
app.add_middleware(MonitoringMiddleware)


@app.exception_handler(RequestValidationError)
async def validation_error(request, error):
    # Do not echo transaction values; this also makes nonfinite-input errors JSON-safe.
    return JSONResponse(status_code=422, content={"detail": [
        {"loc": list(item["loc"]), "msg": item["msg"], "type": item["type"]} for item in error.errors()
    ]})


@app.get("/health")
def health(request: Request):
    pipeline = request.app.state.pipeline
    return {"status": "ok", "model_id": pipeline.metadata["model_id"], "model_status": pipeline.metadata["status"],
            "scenario": pipeline.metadata["economics"]["version"], "threshold": pipeline.threshold,
            "probability_calibrated": False, "release_id": RELEASE_ID,
            "revision": source_revision()}


@app.get("/metrics")
def metrics(request: Request):
    return request.app.state.telemetry.snapshot()


@app.post("/predict", response_model=PredictionResponse)
def predict(body: PredictionRequest, request: Request):
    pipeline = request.app.state.pipeline
    try:
        inputs = pd.DataFrame([item.model_dump() for item in body.transactions])
        rows = pipeline.predict(inputs)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    request.app.state.telemetry.record_predictions(inputs, rows)
    return {"model_id": pipeline.metadata["model_id"], "model_status": pipeline.metadata["status"],
            "scenario": pipeline.metadata["economics"]["version"], "threshold": pipeline.threshold,
            "predictions": rows.to_dict(orient="records")}
