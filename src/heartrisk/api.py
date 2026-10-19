"""FastAPI service: /predict, /health, /model-info and Prometheus /metrics.

Run:  PYTHONPATH=src uvicorn heartrisk.api:app --port 8000
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import __version__
from .predict import load_model, load_model_card, predict_records

# --------------------------------------------------------------------------- logging


class JsonFormatter(logging.Formatter):
    """One JSON object per line: easy to grep, ship to Loki/ELK, or tail."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "extra_fields", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _configure_logging() -> logging.Logger:
    logger = logging.getLogger("heartrisk")
    if logger.handlers:
        return logger
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
    logger.propagate = False
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(JsonFormatter())
    logger.addHandler(stream)
    log_file = os.environ.get("HEARTRISK_LOG_FILE")
    if log_file:
        try:
            fh = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3)
            fh.setFormatter(JsonFormatter())
            logger.addHandler(fh)
        except OSError:  # read-only filesystem: stdout logging is enough
            pass
    return logger


log = _configure_logging()


def _log(message: str, **fields) -> None:
    log.info(message, extra={"extra_fields": fields})


# --------------------------------------------------------------------------- metrics

REQUESTS = Counter("heartrisk_http_requests_total", "HTTP requests",
                   ["method", "endpoint", "status"])
LATENCY = Histogram("heartrisk_http_request_duration_seconds", "Request latency in seconds",
                    ["endpoint"], buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PREDICTIONS = Counter("heartrisk_predictions_total", "Predictions served", ["label"])
PROBABILITY = Histogram("heartrisk_predicted_probability", "Predicted P(heart disease)",
                        buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
MODEL_LOADED = Gauge("heartrisk_model_loaded", "1 if the model is loaded")
IN_FLIGHT = Gauge("heartrisk_in_flight_requests", "Requests currently being handled")

# --------------------------------------------------------------------------- schemas


class PatientFeatures(BaseModel):
    """Input record. Field names and codes follow the UCI Cleveland data set."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {
            "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
            "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
            "ca": 0, "thal": 6,
        }},
    )

    age: int = Field(..., ge=18, le=100, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1 typical angina ... 4 asymptomatic)")
    trestbps: float = Field(..., ge=60, le=250, description="Resting blood pressure, mm Hg")
    chol: float = Field(..., ge=80, le=700, description="Serum cholesterol, mg/dl")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result")
    thalach: float = Field(..., ge=60, le=230, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=1, le=3, description="Slope of the peak exercise ST segment")
    ca: Optional[int] = Field(None, ge=0, le=3, description="Major vessels coloured by fluoroscopy; may be omitted")
    thal: Optional[int] = Field(None, description="3 normal, 6 fixed defect, 7 reversible defect; may be omitted")

    @field_validator("thal")
    @classmethod
    def _thal_codes(cls, v):
        if v is not None and v not in (3, 6, 7):
            raise ValueError("thal must be one of 3, 6, 7")
        return v


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = heart disease predicted, 0 = not predicted")
    label: str
    probability_disease: float = Field(..., description="P(heart disease)")
    confidence: float = Field(..., description="Probability of the predicted class")
    model_version: str


# --------------------------------------------------------------------------- app


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model = load_model()
        app.state.card = load_model_card()
        MODEL_LOADED.set(1)
        _log("model loaded", version=app.state.card.get("version"),
             algorithm=app.state.card.get("algorithm"))
    except FileNotFoundError as exc:
        app.state.model, app.state.card = None, {}
        MODEL_LOADED.set(0)
        log.error("model could not be loaded: %s", exc)
    yield


app = FastAPI(
    title="HeartRisk API",
    version=__version__,
    description="Predicts heart disease risk from patient health data (UCI Cleveland).",
    lifespan=lifespan,
)


@app.middleware("http")
async def observe(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    start = time.perf_counter()
    IN_FLIGHT.inc()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        IN_FLIGHT.dec()
        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        endpoint = getattr(route, "path", "unmatched")
        if endpoint != "/metrics":
            REQUESTS.labels(request.method, endpoint, str(status)).inc()
            LATENCY.labels(endpoint).observe(elapsed)
            _log("request", request_id=request_id, method=request.method, path=request.url.path,
                 status=status, latency_ms=round(elapsed * 1000, 2),
                 client=request.client.host if request.client else None)


@app.get("/health")
def health(request: Request):
    loaded = request.app.state.model is not None
    if not loaded:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model_loaded": True, "api_version": __version__}


@app.get("/model-info")
def model_info(request: Request):
    card = request.app.state.card
    if not card:
        raise HTTPException(status_code=503, detail="model card not available")
    keep = ("model_name", "algorithm", "version", "trained_at", "features", "test_metrics", "top_features")
    return {k: card[k] for k in keep if k in card}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures, request: Request):
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    record = patient.model_dump()
    result = predict_records(model, [record])[0]
    PREDICTIONS.labels(result["label"]).inc()
    PROBABILITY.observe(result["probability_disease"])
    _log("prediction", input=record, **result)
    return PredictionResponse(**result, model_version=request.app.state.card.get("version", "unknown"))


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
