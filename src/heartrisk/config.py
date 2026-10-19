"""Central configuration: paths, column definitions and random seed.

Every other module imports from here so that paths and feature lists are
defined exactly once (training and serving cannot drift apart).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("HEARTRISK_HOME", Path(__file__).resolve().parents[2]))

RAW_PATH = ROOT / "data" / "raw" / "processed.cleveland.data"
CLEAN_PATH = ROOT / "data" / "processed" / "heart_clean.csv"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
MODEL_DIR = ROOT / "models"
MODEL_FILE = "heart_pipeline.joblib"
MODEL_CARD_FILE = "model_card.json"

MLFLOW_DB = ROOT / "mlflow.db"
MLFLOW_ARTIFACTS = ROOT / "mlartifacts"
MLFLOW_EXPERIMENT = "heartrisk-cleveland"
REGISTERED_MODEL_NAME = "heartrisk-classifier"

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Column order of processed.cleveland.data (no header row in the file).
RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
TARGET = "target"

# The 13 model inputs, in the order the API and the pipeline expect them.
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

# Engineered in features.py (inside the pipeline, so serving gets them too).
ENGINEERED = ["hr_ratio", "chol_high", "bp_high"]

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"] + ENGINEERED
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

CLASS_NAMES = {0: "No heart disease", 1: "Heart disease"}


def model_path() -> Path:
    """Model location; overridable so tests and containers can relocate it."""
    return Path(os.environ.get("HEARTRISK_MODEL_PATH", MODEL_DIR / MODEL_FILE))


def model_card_path() -> Path:
    return model_path().with_name(MODEL_CARD_FILE)
