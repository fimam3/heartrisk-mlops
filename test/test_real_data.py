"""Checks against the REAL Cleveland data and the shipped model.

Skipped automatically when the files are absent (e.g. a fresh clone before the
data download), so they never make the offline unit-test run fail.
"""
import json

import pytest

from heartrisk import config, data
from heartrisk.predict import load_model

needs_data = pytest.mark.skipif(not config.RAW_PATH.exists(), reason="raw dataset not downloaded")
needs_model = pytest.mark.skipif(
    not (config.RAW_PATH.exists() and config.model_path().exists() and config.model_card_path().exists()),
    reason="trained model or dataset not available",
)


@needs_data
def test_raw_file_matches_published_cleveland_statistics():
    raw = data.load_raw()
    assert raw.shape == (303, 14)
    assert raw["age"].mean() == pytest.approx(54.44, abs=0.01)
    assert raw["ca"].isna().sum() == 4 and raw["thal"].isna().sum() == 2


@needs_data
def test_clean_class_balance_matches_uci_documentation():
    clean = data.clean(data.load_raw())
    counts = clean[config.TARGET].value_counts().to_dict()
    assert counts == {0: 164, 1: 139}


@needs_model
def test_shipped_model_reproduces_its_recorded_test_metrics():
    from heartrisk.train import score_predictions

    clean = data.clean(data.load_raw())
    _, X_test, _, y_test = data.split_xy(clean)
    proba = load_model().predict_proba(X_test)[:, 1]
    recorded = json.loads(config.model_card_path().read_text())["test_metrics"]
    fresh = score_predictions(y_test, proba)
    for name in ("accuracy", "precision", "recall", "roc_auc"):
        assert fresh[name] == pytest.approx(recorded[name], abs=1e-6)


@needs_model
def test_shipped_model_meets_quality_gate():
    clean = data.clean(data.load_raw())
    _, X_test, _, y_test = data.split_xy(clean)
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y_test, load_model().predict_proba(X_test)[:, 1]) >= 0.80
