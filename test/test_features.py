import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from heartrisk import config
from heartrisk.features import add_clinical_features, build_pipeline, build_preprocessor


def _row(**overrides) -> pd.DataFrame:
    base = {"age": 60, "sex": 1, "cp": 4, "trestbps": 150, "chol": 250, "fbs": 0, "restecg": 0,
            "thalach": 100, "exang": 1, "oldpeak": 1.0, "slope": 2, "ca": 1, "thal": 7}
    base.update(overrides)
    return pd.DataFrame([base], columns=config.FEATURES).astype(float)


def test_engineered_feature_values():
    out = add_clinical_features(_row())
    assert out.loc[0, "hr_ratio"] == 100 / (220 - 60)
    assert out.loc[0, "chol_high"] == 1.0
    assert out.loc[0, "bp_high"] == 1.0


def test_engineered_thresholds_are_inclusive():
    out = add_clinical_features(_row(chol=239, trestbps=140))
    assert out.loc[0, "chol_high"] == 0.0
    assert out.loc[0, "bp_high"] == 1.0


def test_missing_inputs_stay_missing():
    out = add_clinical_features(_row(chol=np.nan, trestbps=np.nan))
    assert np.isnan(out.loc[0, "chol_high"])
    assert np.isnan(out.loc[0, "bp_high"])


def test_add_features_does_not_mutate_input():
    X = _row()
    before = X.copy()
    add_clinical_features(X)
    pd.testing.assert_frame_equal(X, before)


def test_preprocessor_output_has_no_nans(clean_df):
    X = clean_df[config.FEATURES].astype(float)
    Z = build_preprocessor().fit_transform(X)
    assert not np.isnan(np.asarray(Z)).any()
    assert Z.shape[0] == len(X)


def test_scaled_numeric_columns_are_standardised(clean_df):
    X = clean_df[config.FEATURES].astype(float)
    Z = build_preprocessor().fit_transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
    n_num = len(config.NUMERIC)
    assert np.allclose(Z[:, :n_num].mean(axis=0), 0, atol=1e-6)
    assert np.allclose(Z[:, :n_num].std(axis=0), 1, atol=1e-6)


def test_pipeline_handles_unseen_category_and_missing_values(clean_df):
    X, y = clean_df[config.FEATURES].astype(float), clean_df[config.TARGET]
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    odd = _row(cp=9, thal=np.nan, ca=np.nan)  # cp=9 never seen in training
    proba = pipe.predict_proba(odd)[0, 1]
    assert 0.0 <= proba <= 1.0


def test_pipeline_is_deterministic(clean_df):
    X, y = clean_df[config.FEATURES].astype(float), clean_df[config.TARGET]
    a = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y).predict_proba(X)
    b = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y).predict_proba(X)
    assert np.allclose(a, b)
