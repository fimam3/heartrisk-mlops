"""Shared fixtures.

The data generated here is SYNTHETIC and exists only so the unit tests run
without network access. It has the same columns/codes as the Cleveland file but
is never used for training the shipped model or for any reported result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heartrisk import config, data, train


def make_synthetic_raw(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(29, 78, n)
    sex = rng.integers(0, 2, n)
    cp = rng.integers(1, 5, n)
    trestbps = rng.normal(131, 17, n).round()
    chol = rng.normal(246, 50, n).round()
    fbs = (rng.random(n) < 0.15).astype(int)
    restecg = rng.choice([0, 1, 2], n, p=[0.5, 0.01, 0.49])
    thalach = np.clip(rng.normal(150, 22, n) - (age - 54) * 0.8, 70, 202).round()
    exang = (rng.random(n) < 0.33).astype(int)
    oldpeak = np.clip(rng.gamma(1.3, 0.8, n), 0, 6).round(1)
    slope = rng.choice([1, 2, 3], n, p=[0.46, 0.46, 0.08])
    ca = rng.choice([0, 1, 2, 3], n, p=[0.58, 0.22, 0.13, 0.07]).astype(float)
    thal = rng.choice([3, 6, 7], n, p=[0.55, 0.06, 0.39]).astype(float)

    logit = (-3.2 + 0.03 * age + 0.9 * sex + 0.5 * (cp == 4) + 0.015 * (trestbps - 130)
             - 0.025 * (thalach - 150) + 1.0 * exang + 0.6 * oldpeak + 0.8 * ca
             + 0.9 * (thal == 7))
    p = 1 / (1 + np.exp(-logit))
    disease = rng.random(n) < p
    num = np.where(disease, rng.integers(1, 5, n), 0)

    df = pd.DataFrame({
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol, "fbs": fbs,
        "restecg": restecg, "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
        "slope": slope, "ca": ca, "thal": thal, "num": num,
    })
    df.loc[rng.choice(n, 4, replace=False), "ca"] = np.nan
    df.loc[rng.choice(n, 2, replace=False), "thal"] = np.nan
    return df


def write_raw_file(df: pd.DataFrame, path) -> None:
    """Write in the UCI layout: no header, '?' for missing values."""
    df.to_csv(path, header=False, index=False, na_rep="?")


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return make_synthetic_raw()


@pytest.fixture(scope="session")
def clean_df(raw_df) -> pd.DataFrame:
    return data.clean(raw_df)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, clean_df):
    """Run the real training code (small grids) into a temp directory."""
    root = tmp_path_factory.mktemp("run")
    out = train.train(
        clean_df,
        model_dir=root / "models",
        report_dir=root / "reports",
        mlflow_db=root / "mlflow.db",
        mlflow_artifacts=root / "mlartifacts",
        quick=True,
    )
    out["root"] = root
    return out


@pytest.fixture
def sample_patient() -> dict:
    return {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
        "ca": 0, "thal": 6,
    }


assert config.FEATURES  # fail fast if config import is broken
