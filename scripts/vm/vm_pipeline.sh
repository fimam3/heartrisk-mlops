#!/usr/bin/env bash
# Step 2 - run the whole ML pipeline: clean data, EDA, training with MLflow, quality gate, tests, lint.
# Output is also saved to logs/vm_pipeline.log.
#
# Usage:  bash scripts/vm/vm_pipeline.sh            # full run (training takes a few minutes)
#         bash scripts/vm/vm_pipeline.sh --quick    # smaller tuning grid, for a fast rehearsal
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_venv

TRAIN_ARGS=()
[ "${1:-}" = "--quick" ] && TRAIN_ARGS+=(--quick)

run_pipeline() {
  say "1/6  Clean the data"
  python scripts/prepare_data.py

  say "2/6  Exploratory data analysis (figures + MLflow run)"
  python -m heartrisk.eda --log-mlflow

  say "3/6  Train and tune models, track with MLflow"
  python -m heartrisk.train ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}

  say "4/6  Model quality gate (ROC-AUC >= 0.80)"
  python scripts/check_metrics.py --min-auc 0.80

  say "5/6  Unit tests"
  python -m pytest -q

  say "6/6  Lint"
  python -m flake8 . --count --statistics

  say "Pipeline finished successfully"
  echo "Outputs: models/  reports/  reports/figures/  mlflow.db  mlartifacts/"
  echo "MLflow UI:  bash scripts/vm/vm_mlflow.sh"
}

run_pipeline 2>&1 | tee logs/vm_pipeline.log
exit "${PIPESTATUS[0]}"
