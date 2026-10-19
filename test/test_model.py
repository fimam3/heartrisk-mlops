import json

import joblib
import numpy as np
import pytest

from heartrisk import config
from heartrisk.predict import load_model, predict_records
from heartrisk.train import candidates, score_predictions


def test_at_least_two_candidate_models():
    assert len(candidates()) >= 2
    assert {"logistic_regression", "random_forest"} <= set(candidates())


def test_score_predictions_perfect_and_worst():
    y = np.array([0, 0, 1, 1])
    perfect = score_predictions(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert perfect["accuracy"] == 1.0 and perfect["roc_auc"] == 1.0
    worst = score_predictions(y, np.array([0.9, 0.8, 0.2, 0.1]))
    assert worst["accuracy"] == 0.0 and worst["roc_auc"] == 0.0


def test_training_writes_all_artifacts(trained):
    root = trained["root"]
    assert (root / "models" / config.MODEL_FILE).exists()
    assert (root / "models" / config.MODEL_CARD_FILE).exists()
    assert (root / "reports" / "metrics.json").exists()
    assert (root / "reports" / "model_comparison.csv").exists()
    assert (root / "reports" / "figures" / "model_roc_comparison.png").exists()
    assert (root / "mlflow.db").exists()


def test_model_card_contents(trained):
    card = json.loads((trained["root"] / "models" / config.MODEL_CARD_FILE).read_text())
    assert card["features"] == config.FEATURES
    assert set(card["test_metrics"]) >= {"accuracy", "precision", "recall", "roc_auc"}
    assert len(card["version"]) == 12


def test_mlflow_logged_params_metrics_and_artifacts(trained):
    import mlflow

    client = mlflow.MlflowClient()
    run = client.get_run(trained["card"]["best_child_run_id"])
    assert "cv_roc_auc_mean" in run.data.metrics
    assert "test_recall" in run.data.metrics
    assert "model" in run.data.params
    names = {a.path for a in client.list_artifacts(run.info.run_id)}
    assert {"plots", "classification_report.json"} <= names
    versions = client.search_model_versions(f"name='{config.REGISTERED_MODEL_NAME}'")
    assert len(versions) >= 1  # best model was registered


def test_model_learns_signal_on_synthetic_data(trained):
    assert trained["card"]["test_metrics"]["roc_auc"] > 0.70


def test_saved_pipeline_reloads_and_predicts(trained, sample_patient):
    model = load_model(trained["root"] / "models" / config.MODEL_FILE)
    out = predict_records(model, [sample_patient])[0]
    assert out["prediction"] in (0, 1)
    assert 0 <= out["probability_disease"] <= 1
    assert 0.5 <= out["confidence"] <= 1


def test_predict_records_handles_missing_optional_fields(trained, sample_patient):
    model = joblib.load(trained["root"] / "models" / config.MODEL_FILE)
    sample_patient.update(ca=None, thal=None)
    out = predict_records(model, [sample_patient])
    assert len(out) == 1


def test_load_model_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nope.joblib")
