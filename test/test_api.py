import pytest
from fastapi.testclient import TestClient

from heartrisk import config
from heartrisk.api import app


@pytest.fixture
def client(trained, monkeypatch):
    monkeypatch.setenv("HEARTRISK_MODEL_PATH", str(trained["root"] / "models" / config.MODEL_FILE))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_no_model(tmp_path, monkeypatch):
    monkeypatch.setenv("HEARTRISK_MODEL_PATH", str(tmp_path / "missing.joblib"))
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_prediction_and_confidence(client, sample_patient):
    r = client.post("/predict", json=sample_patient)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert 0.0 <= body["probability_disease"] <= 1.0
    assert body["label"] in config.CLASS_NAMES.values()
    assert body["model_version"]
    assert "x-request-id" in r.headers


def test_predict_accepts_missing_optional_fields(client, sample_patient):
    del sample_patient["ca"], sample_patient["thal"]
    assert client.post("/predict", json=sample_patient).status_code == 200


@pytest.mark.parametrize("field,value", [
    ("age", 5), ("sex", 2), ("cp", 0), ("trestbps", 1000), ("thal", 5), ("oldpeak", -1),
])
def test_predict_rejects_out_of_range_values(client, sample_patient, field, value):
    sample_patient[field] = value
    assert client.post("/predict", json=sample_patient).status_code == 422


def test_predict_rejects_missing_required_field(client, sample_patient):
    del sample_patient["age"]
    assert client.post("/predict", json=sample_patient).status_code == 422


def test_predict_rejects_unknown_field(client, sample_patient):
    sample_patient["favourite_colour"] = "blue"
    assert client.post("/predict", json=sample_patient).status_code == 422


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["features"] == config.FEATURES
    assert "test_metrics" in body


def test_metrics_endpoint_exposes_prometheus_counters(client, sample_patient):
    client.post("/predict", json=sample_patient)
    text = client.get("/metrics").text
    assert "heartrisk_http_requests_total" in text
    assert "heartrisk_predictions_total" in text
    assert "heartrisk_http_request_duration_seconds_bucket" in text
    assert "heartrisk_model_loaded 1.0" in text


def test_service_reports_unavailable_without_model(client_no_model, sample_patient):
    assert client_no_model.get("/health").status_code == 503
    assert client_no_model.post("/predict", json=sample_patient).status_code == 503
