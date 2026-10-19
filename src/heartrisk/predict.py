"""Load the saved pipeline and turn patient records into predictions."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from . import config


def load_model(path: Path | str | None = None):
    path = Path(path) if path else config.model_path()
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}. Run `python -m heartrisk.train` first.")
    return joblib.load(path)


@lru_cache(maxsize=1)
def _cached_card(path: str) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def load_model_card(path: Path | str | None = None) -> dict:
    return dict(_cached_card(str(path or config.model_card_path())))


def predict_records(model, records: list[dict]) -> list[dict]:
    """Predict for a list of feature dicts.

    ``confidence`` is the probability of the *predicted* class;
    ``probability_disease`` is always P(class = 1).
    """
    frame = pd.DataFrame(records, columns=config.FEATURES).astype(float)
    proba = model.predict_proba(frame)[:, 1]
    out = []
    for p in proba:
        pred = int(p >= 0.5)
        out.append({
            "prediction": pred,
            "label": config.CLASS_NAMES[pred],
            "probability_disease": round(float(p), 4),
            "confidence": round(float(p if pred == 1 else 1 - p), 4),
        })
    return out
