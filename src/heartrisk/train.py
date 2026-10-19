"""Model selection, tuning, evaluation and MLflow tracking.

Run:  PYTHONPATH=src python -m heartrisk.train [--quick]

Method
------
1. Stratified 80/20 train/test split (test set is touched once, at the end).
2. For each candidate (logistic regression, random forest, gradient boosting):
   * **nested cross-validation** on the training set (5 outer folds, 3 inner
     folds for the grid search) to get an unbiased estimate of accuracy,
     precision, recall, F1 and ROC-AUC for "tune then evaluate";
   * a final grid search (5 folds, ROC-AUC) on the whole training set to pick
     hyper-parameters, then evaluation on the held-out test set.
3. The candidate with the best nested-CV ROC-AUC is saved as the deployable
   pipeline (preprocessing + classifier) and registered in MLflow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from . import config, data
from .features import build_pipeline
from .plotting import INK_SECONDARY, MODEL_COLORS, apply_style
from .tracking import setup_mlflow

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def candidates(quick: bool = False) -> dict:
    """Estimator + hyper-parameter grid per model (``model__`` = pipeline step)."""
    rs = config.RANDOM_STATE
    if quick:  # tiny grids so the unit tests stay fast
        return {
            "logistic_regression": (
                LogisticRegression(max_iter=2000),
                {"model__C": [0.1, 1.0]},
            ),
            "random_forest": (
                RandomForestClassifier(random_state=rs, n_jobs=1),
                {"model__n_estimators": [50], "model__max_depth": [3, None]},
            ),
        }
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000),
            {"model__C": [0.01, 0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=rs, n_jobs=1),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [3, 5, None],
                "model__min_samples_leaf": [1, 3, 5],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=rs),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
    }


def score_predictions(y_true, proba) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, cwd=config.ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def _clean_params(params: dict) -> dict:
    return {k.replace("model__", ""): ("None" if v is None else v) for k, v in params.items()}


def evaluate_candidate(name, estimator, grid, X_train, X_test, y_train, y_test):
    pipe = build_pipeline(estimator)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE + 1)

    search_for_cv = GridSearchCV(clone(pipe), grid, cv=inner, scoring="roc_auc", n_jobs=-1)
    cv = cross_validate(search_for_cv, X_train, y_train, cv=outer, scoring=SCORING, n_jobs=1)

    search = GridSearchCV(
        clone(pipe), grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE),
        scoring="roc_auc", n_jobs=-1,
    )
    t0 = time.perf_counter()
    search.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0

    best = search.best_estimator_
    proba = best.predict_proba(X_test)[:, 1]
    return {
        "name": name,
        "pipeline": best,
        "best_params": _clean_params(search.best_params_),
        "tuning_cv_roc_auc": float(search.best_score_),
        "cv": {m: (float(cv[f"test_{m}"].mean()), float(cv[f"test_{m}"].std())) for m in SCORING},
        "test": {k: float(v) for k, v in score_predictions(y_test, proba).items()},
        "proba": proba,
        "fit_seconds": fit_seconds,
        "n_candidates": len(search.cv_results_["params"]),
    }


def plot_roc(result, y_test, out: Path) -> None:
    fpr, tpr, _ = roc_curve(y_test, result["proba"])
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot(fpr, tpr, color=MODEL_COLORS[0], linewidth=2)
    ax.plot([0, 1], [0, 1], color=INK_SECONDARY, linewidth=1, linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC - {result['name']} (AUC {result['test']['roc_auc']:.3f})")
    fig.savefig(out)
    plt.close(fig)


def plot_confusion(result, y_test, out: Path) -> None:
    pred = (result["proba"] >= 0.5).astype(int)
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["No disease", "Disease"], cmap="Blues", ax=ax, colorbar=False
    )
    ax.grid(False)
    ax.set_title(f"Confusion matrix - {result['name']}")
    fig.savefig(out)
    plt.close(fig)


def plot_roc_overlay(results, y_test, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for color, r in zip(MODEL_COLORS, results):
        fpr, tpr, _ = roc_curve(y_test, r["proba"])
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{r['name']} (AUC {r['test']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], color=INK_SECONDARY, linewidth=1, linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Held-out test ROC curves")
    ax.legend(loc="lower right")
    fig.savefig(out)
    plt.close(fig)


def plot_importance(best, X_test, y_test, out: Path) -> pd.DataFrame:
    res = permutation_importance(best, X_test, y_test, scoring="roc_auc",
                                 n_repeats=30, random_state=config.RANDOM_STATE)
    imp = pd.DataFrame({"feature": X_test.columns, "importance": res.importances_mean,
                        "std": res.importances_std}).sort_values("importance")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(imp["feature"], imp["importance"], xerr=imp["std"], color=MODEL_COLORS[0],
            error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1})
    ax.set_xlabel("Drop in test ROC-AUC when the feature is shuffled")
    ax.set_title("Permutation importance (final model)")
    ax.grid(axis="y", visible=False)
    fig.savefig(out)
    plt.close(fig)
    return imp.sort_values("importance", ascending=False)


def train(
    df: pd.DataFrame,
    *,
    model_dir: Path = config.MODEL_DIR,
    report_dir: Path = config.REPORT_DIR,
    mlflow_db: Path = config.MLFLOW_DB,
    mlflow_artifacts: Path = config.MLFLOW_ARTIFACTS,
    data_path: Path | None = None,
    quick: bool = False,
    register: bool = True,
) -> dict:
    apply_style()
    model_dir, report_dir = Path(model_dir), Path(report_dir)
    fig_dir = report_dir / "figures"
    for d in (model_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = data.split_xy(df)
    exp_id = setup_mlflow(mlflow_db, mlflow_artifacts)

    results = []
    with mlflow.start_run(experiment_id=exp_id, run_name="model-selection") as parent:
        mlflow.set_tags({"stage": "training", "git_sha": _git_sha()})
        mlflow.log_params({
            "n_rows": len(df), "n_train": len(X_train), "n_test": len(X_test),
            "test_size": config.TEST_SIZE, "random_state": config.RANDOM_STATE,
            "cv_outer_folds": 5, "cv_inner_folds": 3, "quick_mode": quick,
            "sklearn_version": sklearn.__version__, "python_version": platform.python_version(),
        })
        if data_path and Path(data_path).exists():
            mlflow.log_param("data_sha256", _sha256(Path(data_path))[:16])

        for name, (estimator, grid) in candidates(quick).items():
            r = evaluate_candidate(name, estimator, grid, X_train, X_test, y_train, y_test)
            print(f"[{name}] nested-CV AUC {r['cv']['roc_auc'][0]:.3f} | test AUC {r['test']['roc_auc']:.3f}"
                  f" | best params {r['best_params']}")
            with mlflow.start_run(experiment_id=exp_id, run_name=name, nested=True) as child:
                r["run_id"] = child.info.run_id
                mlflow.set_tags({"stage": "training", "model_family": name})
                mlflow.log_param("model", name)
                mlflow.log_params(r["best_params"])
                mlflow.log_param("grid_size", r["n_candidates"])
                mlflow.log_metric("tuning_cv_roc_auc", r["tuning_cv_roc_auc"])
                mlflow.log_metric("fit_seconds", r["fit_seconds"])
                for m, (mean, std) in r["cv"].items():
                    mlflow.log_metric(f"cv_{m}_mean", mean)
                    mlflow.log_metric(f"cv_{m}_std", std)
                for m, v in r["test"].items():
                    mlflow.log_metric(f"test_{m}", v)

                roc_path, cm_path = fig_dir / f"roc_{name}.png", fig_dir / f"confusion_{name}.png"
                plot_roc(r, y_test, roc_path)
                plot_confusion(r, y_test, cm_path)
                mlflow.log_artifact(str(roc_path), "plots")
                mlflow.log_artifact(str(cm_path), "plots")
                rep = classification_report(y_test, (r["proba"] >= 0.5).astype(int),
                                            target_names=["no_disease", "disease"], output_dict=True)
                mlflow.log_dict(rep, "classification_report.json")
                sig = infer_signature(X_train, r["pipeline"].predict_proba(X_train)[:, 1])
                mlflow.sklearn.log_model(
                    r["pipeline"], name="model", signature=sig, input_example=X_train.head(3),
                    # our own trusted artifact; the pipeline contains a custom function
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                )
            results.append(r)

        best = max(results, key=lambda r: r["cv"]["roc_auc"][0])
        mlflow.set_tag("best_model", best["name"])
        mlflow.log_metric("best_cv_roc_auc", best["cv"]["roc_auc"][0])
        mlflow.log_metric("best_test_roc_auc", best["test"]["roc_auc"])

        comparison = pd.DataFrame([
            {"model": r["name"],
             **{f"cv_{m}": f"{mean:.3f} +/- {std:.3f}" for m, (mean, std) in r["cv"].items()},
             **{f"test_{m}": round(v, 3) for m, v in r["test"].items()},
             "best_params": json.dumps(r["best_params"])}
            for r in results
        ])
        comparison_path = report_dir / "model_comparison.csv"
        comparison.to_csv(comparison_path, index=False)

        overlay_path = fig_dir / "model_roc_comparison.png"
        plot_roc_overlay(results, y_test, overlay_path)
        cm_best_path = fig_dir / "confusion_best_model.png"
        plot_confusion(best, y_test, cm_best_path)
        imp = plot_importance(best["pipeline"], X_test, y_test, fig_dir / "feature_importance.png")

        model_path = model_dir / config.MODEL_FILE
        joblib.dump(best["pipeline"], model_path)

        card = {
            "model_name": config.REGISTERED_MODEL_NAME,
            "algorithm": best["name"],
            "version": _sha256(model_path)[:12],
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "features": config.FEATURES,
            "target": "1 = heart disease present, 0 = absent",
            "best_params": best["best_params"],
            "cv_metrics": {m: {"mean": v[0], "std": v[1]} for m, v in best["cv"].items()},
            "test_metrics": best["test"],
            "top_features": imp.head(5)["feature"].tolist(),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
            "git_sha": _git_sha(),
            "mlflow_run_id": parent.info.run_id,
            "best_child_run_id": best["run_id"],
        }
        card_path = model_dir / config.MODEL_CARD_FILE
        card_path.write_text(json.dumps(card, indent=2))
        metrics_path = report_dir / "metrics.json"
        metrics_path.write_text(json.dumps({
            "best_model": best["name"],
            "cv_roc_auc": best["cv"]["roc_auc"][0],
            "test": best["test"],
            "all_models": {r["name"]: {"cv_roc_auc": r["cv"]["roc_auc"][0], "test": r["test"]}
                           for r in results},
        }, indent=2))

        for p in (comparison_path, overlay_path, cm_best_path, fig_dir / "feature_importance.png",
                  card_path, metrics_path):
            mlflow.log_artifact(str(p), "summary")
        if data_path and Path(data_path).exists():
            mlflow.log_artifact(str(data_path), "dataset")

        if register:
            mv = mlflow.register_model(f"runs:/{best['run_id']}/model", config.REGISTERED_MODEL_NAME)
            mlflow.set_tag("registered_model_version", mv.version)

    print(f"\nBest model: {best['name']}  nested-CV AUC={best['cv']['roc_auc'][0]:.3f}  "
          f"test AUC={best['test']['roc_auc']:.3f}")
    print("Saved:", model_path)
    return {"best": best["name"], "card": card, "model_path": model_path, "results": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input", default=str(config.CLEAN_PATH), help="cleaned CSV")
    parser.add_argument("--quick", action="store_true", help="small grids (smoke test)")
    parser.add_argument("--no-register", action="store_true", help="skip the MLflow model registry")
    args = parser.parse_args(argv)

    path = Path(args.input)
    if not path.exists():
        print(f"ERROR: {path} not found. Run scripts/prepare_data.py first.", file=sys.stderr)
        return 2
    df = data.load_clean(path)
    train(df, data_path=path, quick=args.quick, register=not args.no_register)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
