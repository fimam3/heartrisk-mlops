# HeartRisk MLOps

End-to-end MLOps pipeline that predicts the risk of heart disease from patient health data
(UCI Heart Disease, Cleveland subset) and serves it as a containerised, monitored API.

> Course: MLOps (AIMLCZG523), Assignment 01. Repository link: `<add your GitHub URL here>`

```
 UCI data ──► prepare_data ──► EDA ──► train (nested CV + grid search) ──► MLflow (runs, registry)
                                                   │
                                                   ▼
                                  models/heart_pipeline.joblib  (preprocessing + classifier)
                                                   │
        GitHub Actions: lint → pytest → train → quality gate → docker build + smoke test
                                                   │
                                                   ▼
        Docker image ──► Kubernetes (Deployment, LoadBalancer Service, Ingress, HPA)
                                                   │
                      FastAPI  /predict  /health  /model-info  /metrics
                                                   │
                  JSON request logs ──► Prometheus ──► Grafana dashboard
```

## 1. Setup

Python 3.13 is what this was tested with.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=src                                   # Windows PowerShell: $env:PYTHONPATH="src"
```

### 1a. Tools to install

| Tool | Needed for | Install |
|---|---|---|
| Git + a GitHub account | repository, GitHub Actions | https://git-scm.com/downloads (Windows: `winget install Git.Git`) |
| Python 3.13 | training, tests, API | https://www.python.org/downloads/ (or `uv python install 3.13`) |
| Docker | container, monitoring stack | Docker Desktop (Windows/macOS) or `curl -fsSL https://get.docker.com \| sudo sh` (Linux) |
| kubectl + Kubernetes | deployment | enable Kubernetes in Docker Desktop, or install Minikube (`minikube start --driver=docker`) |
| Prometheus + Grafana | monitoring | nothing to install, they run through `docker compose` (section 8) |

On a Linux VM, `scripts/vm/` contains ready-made scripts that install the tools and run every step in order
(see `scripts/vm/README.md`): `vm_setup.sh`, `vm_pipeline.sh`, `vm_docker.sh`, `vm_k8s.sh`, `vm_monitoring.sh`.

## 2. Run the pipeline

| Step | Command | Output |
|---|---|---|
| Get data | `python scripts/download_data.py` | `data/raw/processed.cleveland.data` |
| Clean data | `python scripts/prepare_data.py` | `data/processed/heart_clean.csv` |
| EDA | `python -m heartrisk.eda --log-mlflow` | `reports/figures/eda_*.png`, `reports/eda_summary.json` |
| Train | `python -m heartrisk.train` | `models/`, `reports/metrics.json`, MLflow runs |
| Quality gate | `python scripts/check_metrics.py --min-auc 0.80` | exit code 1 if below gate |
| Tests | `pytest -q` | 43 tests |
| Lint | `flake8 .` | |
| MLflow UI | `mlflow ui --backend-store-uri sqlite:///mlflow.db` | http://localhost:5000 |

`make all` runs lint, tests, data, EDA and training in order. If the UCI server is unreachable,
download `processed.cleveland.data` from https://archive.ics.uci.edu/dataset/45/heart+disease and
put it in `data/raw/`.

## 3. Design choices

**Data.** The 303-row Cleveland file; the original 0-4 diagnosis is binarised (0 = absent, 1 = present).
`ca` (4 rows) and `thal` (2 rows) contain `?`; they stay missing in the cleaned CSV and are imputed
**inside the sklearn pipeline**, so the imputer only ever learns from training folds (no leakage) and the
API can accept requests with those two fields omitted.

**Features.** Numeric columns: median impute + standardise. Categorical columns (`sex, cp, fbs, restecg, exang,
slope, thal`): mode impute + one-hot (unseen categories are ignored at serving time). Three engineered features,
computed inside the pipeline so training and serving cannot drift:

- `hr_ratio = thalach / (220 - age)`: share of the age-predicted maximum heart rate reached under stress
- `chol_high`: cholesterol >= 240 mg/dl
- `bp_high`: resting systolic BP >= 140 mm Hg

**Models and tuning.** Logistic regression, random forest and gradient boosting, each tuned by grid search on
ROC-AUC. To avoid an optimistic estimate from tuning on the same folds, performance is measured with
**nested cross-validation** (5 outer x 3 inner folds) on the training split, then confirmed once on a held-out
20 % test split. Metrics: accuracy, precision, recall, F1, ROC-AUC. The model with the best nested-CV ROC-AUC
is saved and registered. Because this is a screening problem, recall on the disease class is reported
alongside AUC rather than hidden behind accuracy.

**Experiment tracking.** One MLflow *parent* run (`model-selection`) with one *child* run per model. Each run logs
parameters, CV mean/std and test metrics, ROC and confusion-matrix plots, a classification report and the fitted
pipeline. The parent also logs the comparison table, importance plot, model card and cleaned dataset. The winning
model is registered as `heartrisk-classifier`. An `eda` run holds the EDA figures.

**Packaging.** `models/heart_pipeline.joblib` is the full `Pipeline` (engineering + imputation + scaling +
encoding + classifier); `models/model_card.json` records version, metrics, library versions and the MLflow run id.
Dependencies are pinned in `requirements.txt` (full) and `requirements-serve.txt` (inference only).

## 4. API

```bash
PYTHONPATH=src uvicorn heartrisk.api:app --port 8000        # docs at http://localhost:8000/docs
```

```bash
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
  "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6 }'
```

Response: `{"prediction": 0|1, "label": "...", "probability_disease": 0.xx, "confidence": 0.xx, "model_version": "..."}`
where `confidence` is the probability of the predicted class. Inputs are range-checked (HTTP 422 on bad input);
unknown fields are rejected. Other endpoints: `GET /health`, `GET /model-info`, `GET /metrics`.

## 5. Docker

```bash
docker build -t heartrisk-api:1.0 .
docker run --rm -p 8000:8000 heartrisk-api:1.0
curl localhost:8000/health
```

The image uses `requirements-serve.txt` only, runs as a non-root user and has a health check. The model file
is baked into the image, so commit `models/` after training.

## 6. CI/CD (GitHub Actions: `.github/workflows/ci.yml`)

Job 1: install, **flake8**, **pytest** (JUnit XML), fetch + clean data, EDA, train with MLflow, **quality gate**
(fails below ROC-AUC 0.80), then uploads `ci-artifacts/` logs, models, reports, `mlflow.db` and `mlartifacts/`.
Job 2: downloads the trained model, builds the Docker image, starts it and calls `/health` and `/predict`,
uploading the response and container log. Any lint, test, training or smoke-test failure stops the pipeline.

## 7. Kubernetes

```bash
minikube start
eval $(minikube docker-env)               # build into Minikube's Docker (Docker Desktop: skip this line)
docker build -t heartrisk-api:1.0 .
kubectl apply -f k8s/namespace.yaml && kubectl apply -f k8s/
kubectl -n heartrisk get pods,svc
minikube tunnel                           # separate terminal: gives the LoadBalancer an address
curl http://<EXTERNAL-IP>/health
```

`k8s/` contains the namespace, a 2-replica Deployment (probes, resource limits, non-root, read-only filesystem),
a LoadBalancer Service, an optional Ingress and an HPA. For a managed cluster (GKE/EKS/AKS) push the image to a
registry and change `image:` in `k8s/deployment.yaml`.

## 8. Monitoring and logging

- **Logs:** every request and prediction is written to stdout as one JSON line (request id, path, status,
  latency, input features, output). Inspect with `docker logs <container>` or `kubectl -n heartrisk logs deploy/heartrisk-api`.
- **Metrics:** `/metrics` exposes request counts, latency histograms, predictions by outcome, a histogram of
  predicted probabilities (a simple drift signal), in-flight requests and a model-loaded gauge.
- **Local dashboard:** `docker compose -f docker-compose.monitoring.yml up --build`, then Grafana at
  http://localhost:3000 (admin / admin), pre-provisioned with the *HeartRisk API* dashboard.
- **In-cluster:** `k8s/monitoring/monitoring.yaml` (commands in the file header).

## 9. Repository layout

```
src/heartrisk/    config, data, features, eda, train, predict, api, plotting, tracking
scripts/          download_data, prepare_data, check_metrics
test/             pytest suite (data, features, model + MLflow, API)
data/             raw/ and processed/ (cleaned dataset)
models/           saved pipeline + model card
reports/          figures, metrics, EDA summary, written report
k8s/              Kubernetes manifests (+ monitoring/)
monitoring/       Prometheus config and Grafana dashboard
.github/workflows CI pipeline
screenshots/      evidence for the report
```

## 10. Report and evidence

`reports/` holds the figures (`reports/figures/`), `metrics.json`, `model_comparison.csv`, `eda_summary.json` and
`api_demo_transcript.txt` (real API calls). `screenshots/` holds MLflow screenshots taken from the real run; add your
own CI, Docker, Kubernetes and Grafana screenshots there. `sample.json` is a ready request body:
`curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @sample.json`.

## 11. Tests

The unit tests use a small **synthetic** dataset generated in `test/conftest.py` so they run offline in CI.
It is never used for the shipped model or any reported number; the model is trained only on the real UCI data.
