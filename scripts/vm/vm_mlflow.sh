#!/usr/bin/env bash
# Open the MLflow tracking UI on port 5000 (Ctrl+C to stop).
# Run vm_pipeline.sh first so there are runs to look at.
#
# Usage:  bash scripts/vm/vm_mlflow.sh
# On a headless VM, from your laptop:  ssh -L 5000:localhost:5000 <user>@<vm>   then open http://localhost:5000
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_venv

[ -f mlflow.db ] || die "mlflow.db not found. Run first: bash scripts/vm/vm_pipeline.sh"
say "MLflow UI at http://localhost:5000  (Ctrl+C to stop)"
exec mlflow ui --backend-store-uri "sqlite:///$ROOT/mlflow.db" --host 0.0.0.0 --port 5000
