"""Feature engineering and the preprocessing pipeline.

All transformations live inside one sklearn ``Pipeline`` that is saved together
with the classifier, so the API applies exactly the same steps as training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config

CHOL_HIGH_MG_DL = 240   # "high" total cholesterol threshold
BP_HIGH_MM_HG = 140     # stage-2 hypertension threshold for resting systolic BP


def add_clinical_features(X: pd.DataFrame) -> pd.DataFrame:
    """Append three clinically motivated features.

    * ``hr_ratio``  - peak heart rate achieved / age-predicted maximum (220-age).
      A low ratio under exercise stress is a known warning sign.
    * ``chol_high`` - 1 if cholesterol >= 240 mg/dl.
    * ``bp_high``   - 1 if resting systolic blood pressure >= 140 mm Hg.

    Missing inputs stay missing (NaN) rather than silently becoming 0.
    """
    X = X.copy()
    X["hr_ratio"] = X["thalach"] / (220.0 - X["age"])
    X["chol_high"] = np.where(X["chol"].isna(), np.nan, (X["chol"] >= CHOL_HIGH_MG_DL).astype(float))
    X["bp_high"] = np.where(X["trestbps"].isna(), np.nan, (X["trestbps"] >= BP_HIGH_MM_HG).astype(float))
    return X


def build_preprocessor() -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    columns = ColumnTransformer(
        [
            ("num", numeric, config.NUMERIC),
            ("cat", categorical, config.CATEGORICAL),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("clinical", FunctionTransformer(add_clinical_features, validate=False)),
        ("columns", columns),
    ])


def build_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Preprocessing + classifier as a single deployable object."""
    return Pipeline([
        ("prep", build_preprocessor()),
        ("model", estimator),
    ])
