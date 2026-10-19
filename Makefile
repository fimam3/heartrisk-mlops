PY ?= python
export PYTHONPATH := src

.PHONY: install data eda train test lint serve docker-build docker-run mlflow-ui monitoring k8s-deploy all

install:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) scripts/download_data.py --skip-if-exists
	$(PY) scripts/prepare_data.py

eda:
	$(PY) -m heartrisk.eda --log-mlflow

train:
	$(PY) -m heartrisk.train
	$(PY) scripts/check_metrics.py --min-auc 0.80

lint:
	$(PY) -m flake8 .

test:
	$(PY) -m pytest -q

serve:
	$(PY) -m uvicorn heartrisk.api:app --port 8000 --reload

mlflow-ui:
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

docker-build:
	docker build -t heartrisk-api:1.0 .

docker-run:
	docker run --rm -p 8000:8000 heartrisk-api:1.0

monitoring:
	docker compose -f docker-compose.monitoring.yml up --build

k8s-deploy:
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/

all: lint test data eda train
