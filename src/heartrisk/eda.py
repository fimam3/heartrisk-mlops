"""Exploratory data analysis: figures + a JSON summary.

Run:  PYTHONPATH=src python -m heartrisk.eda [--log-mlflow]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, data
from .features import add_clinical_features
from .plotting import CLASS_COLORS, DIVERGING, INK, INK_SECONDARY, apply_style

HIST_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak", "hr_ratio"]
HIST_TITLES = {
    "age": "Age (years)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "thalach": "Max heart rate achieved (bpm)",
    "oldpeak": "ST depression (exercise vs rest)",
    "hr_ratio": "Heart-rate ratio (peak / (220 - age))",
}
CAT_TITLES = {
    "sex": "Sex (0 = female, 1 = male)",
    "cp": "Chest pain type (1-4)",
    "fbs": "Fasting sugar > 120 (0/1)",
    "restecg": "Resting ECG (0-2)",
    "exang": "Exercise angina (0/1)",
    "slope": "ST slope (1-3)",
    "thal": "Thalassemia (3 / 6 / 7)",
}


def plot_class_balance(df: pd.DataFrame, out: Path) -> None:
    counts = df[config.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.6))
    bars = ax.bar(
        [config.CLASS_NAMES[i] for i in counts.index],
        counts.values,
        color=[CLASS_COLORS[i] for i in counts.index],
        width=0.55,
    )
    total = counts.sum()
    for bar, n in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, n + total * 0.01,
                f"{n}  ({n / total:.0%})", ha="center", va="bottom", color=INK)
    ax.set_ylabel("Patients")
    ax.set_title("Class balance")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.grid(axis="x", visible=False)
    fig.savefig(out)
    plt.close(fig)


def plot_histograms(df: pd.DataFrame, out: Path) -> None:
    feats = add_clinical_features(df[config.FEATURES].astype(float))
    feats[config.TARGET] = df[config.TARGET].values
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    for ax, col in zip(axes.ravel(), HIST_COLUMNS):
        bins = np.histogram_bin_edges(feats[col].dropna(), bins=20)
        for cls in (0, 1):
            ax.hist(feats.loc[feats[config.TARGET] == cls, col].dropna(), bins=bins,
                    color=CLASS_COLORS[cls], alpha=0.55, edgecolor=CLASS_COLORS[cls],
                    linewidth=1.0, label=config.CLASS_NAMES[cls])
        ax.set_title(HIST_TITLES[col], fontsize=10)
        ax.set_ylabel("Patients")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.02))
    fig.suptitle("Distributions by diagnosis", x=0.01, ha="left", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_correlation(df: pd.DataFrame, out: Path) -> None:
    corr = df.corr(numeric_only=True)
    order = corr[config.TARGET].abs().sort_values(ascending=False).index
    corr = corr.loc[order, order]
    fig, ax = plt.subplots(figsize=(9, 7.4))
    im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color=INK if abs(v) < 0.6 else "white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Pearson correlation")
    ax.set_title("Correlation heatmap (sorted by strength vs target)")
    fig.savefig(out)
    plt.close(fig)


def plot_categorical(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4))
    axes = axes.ravel()
    overall = df[config.TARGET].mean()
    for ax, col in zip(axes, config.CATEGORICAL):
        grp = df.groupby(col)[config.TARGET].agg(["mean", "size"])
        labels = [str(int(i)) for i in grp.index]
        bars = ax.bar(labels, grp["mean"], color=CLASS_COLORS[1], width=0.6)
        for bar, n in zip(bars, grp["size"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"n={n}", ha="center", fontsize=8, color=INK_SECONDARY)
        ax.axhline(overall, color=INK_SECONDARY, linewidth=1, linestyle="--")
        ax.set_ylim(0, 1.15)
        ax.set_title(CAT_TITLES[col], fontsize=9.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Share with heart disease")
    axes[4].set_ylabel("Share with heart disease")
    ax = axes[-1]
    miss = df[config.FEATURES].isna().sum()
    miss = miss[miss > 0]
    ax.bar(miss.index, miss.values, color=INK_SECONDARY, width=0.5)
    for i, v in enumerate(miss.values):
        ax.text(i, v + 0.1, str(int(v)), ha="center", fontsize=9, color=INK)
    ax.set_title("Missing values (before imputation)", fontsize=9.5)
    ax.set_ylim(0, max(1, miss.max() * 1.3 if len(miss) else 1))
    ax.grid(axis="x", visible=False)
    fig.suptitle("Disease rate by category (dashed line = overall rate)",
                 x=0.01, ha="left", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def summarise(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "class_counts": {config.CLASS_NAMES[int(k)]: int(v)
                         for k, v in df[config.TARGET].value_counts().sort_index().items()},
        "missing_values": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
        "numeric_summary": df[["age", "trestbps", "chol", "thalach", "oldpeak"]]
        .describe().round(2).to_dict(),
        "top_correlations_with_target": df.corr(numeric_only=True)[config.TARGET]
        .drop(config.TARGET).sort_values(key=abs, ascending=False).head(6).round(3).to_dict(),
    }


def run(df: pd.DataFrame, fig_dir: Path = config.FIG_DIR, report_dir: Path = config.REPORT_DIR) -> list[Path]:
    apply_style()
    fig_dir, report_dir = Path(fig_dir), Path(report_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "eda_class_balance.png": plot_class_balance,
        "eda_histograms.png": plot_histograms,
        "eda_correlation_heatmap.png": plot_correlation,
        "eda_categorical_vs_target.png": plot_categorical,
    }
    paths = []
    for name, fn in outputs.items():
        fn(df, fig_dir / name)
        paths.append(fig_dir / name)
    summary_path = report_dir / "eda_summary.json"
    summary_path.write_text(json.dumps(summarise(df), indent=2))
    paths.append(summary_path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(config.CLEAN_PATH))
    parser.add_argument("--log-mlflow", action="store_true", help="also log the figures to MLflow")
    args = parser.parse_args()

    df = data.load_clean(args.input)
    paths = run(df)
    for p in paths:
        print("wrote", p)

    if args.log_mlflow:
        import mlflow

        from .tracking import setup_mlflow

        exp_id = setup_mlflow()
        with mlflow.start_run(experiment_id=exp_id, run_name="eda"):
            mlflow.set_tag("stage", "eda")
            mlflow.log_params({"rows": len(df), "n_features": len(config.FEATURES)})
            for p in paths:
                mlflow.log_artifact(str(p), artifact_path="eda")
        print("logged EDA artifacts to MLflow")


if __name__ == "__main__":
    main()
