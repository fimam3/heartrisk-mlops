"""Data loading, cleaning and splitting for the Cleveland heart disease data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_raw(path: Path | str = config.RAW_PATH) -> pd.DataFrame:
    """Read the headerless UCI file; '?' marks missing values."""
    return pd.read_csv(path, header=None, names=config.RAW_COLUMNS, na_values="?")


def validate_schema(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return the analysis-ready table.

    * coerce every column to numeric (bad tokens become NaN)
    * drop exact duplicate rows
    * binarise the original 0-4 diagnosis ``num`` into ``target`` (0 = absent,
      1 = present), as the assignment asks for a binary label
    * keep remaining NaNs (``ca`` and ``thal``); they are imputed *inside* the
      sklearn pipeline so the imputer is fitted on training folds only
    """
    validate_schema(df, config.RAW_COLUMNS)
    out = df[config.RAW_COLUMNS].apply(pd.to_numeric, errors="coerce")
    out = out.dropna(subset=["num"]).drop_duplicates().reset_index(drop=True)
    out[config.TARGET] = (out["num"] > 0).astype(int)
    out = out.drop(columns="num")
    return out


def save_clean(df: pd.DataFrame, path: Path | str = config.CLEAN_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_clean(path: Path | str = config.CLEAN_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame):
    """Stratified train/test split. Returns X_train, X_test, y_train, y_test."""
    validate_schema(df, config.FEATURES + [config.TARGET])
    X, y = df[config.FEATURES].astype(float), df[config.TARGET]
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
