# Copyright (c) 2026 Le Thanh Loi. All rights reserved.
# Author: Le Thanh Loi

"""Build publication-style Chapter 4 figures from saved CompPhish outputs.

This script never fits a model.  It reads the row-level Test probabilities and
the metrics produced by ``train_compphish_models.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from compphish_common import PROJECT_AUTHOR, PROJECT_REPOSITORY, project_path


COLORS = {
    "brand_url": "#377eb8",
    "html": "#4daf4a",
    "full": "#e68632",
    "stack": "#7b5ea7",
    "neutral": "#5c677d",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        default="artifacts/compphish/branch_stacking_full",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/compphish/chapter4_figures",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def model_label(model: str) -> str:
    labels = {
        "brand_url__catboost": "Brand–URL CatBoost",
        "html__catboost": "HTML CatBoost",
        "full__catboost": "Full CatBoost",
        "full__xgboost": "Full XGBoost",
        "full__lightgbm": "Full LightGBM",
        "full__extra_trees": "Full Extra Trees",
        "full__stacking_boosting": "Full Boosting Stacking",
        "full__stacking_heterogeneous": "Full Heterogeneous Stacking",
    }
    return labels.get(model, model.replace("__", " – ").replace("_", " ").title())


def score_column(model: str) -> str:
    return f"score__{model}"


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.labelsize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def save(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def test_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics[metrics["split"].eq("test")].copy()
    if result.empty:
        raise ValueError("metrics.csv contains no Test rows")
    return result.set_index("model", drop=False)


def plot_feature_views(metrics: pd.DataFrame, output: Path, dpi: int) -> None:
    models = ["brand_url__catboost", "html__catboost", "full__catboost"]
    measures = [("f1", "F1"), ("pr_auc", "PR-AUC"), ("mcc", "MCC")]
    subset = metrics.loc[models]
    x = np.arange(len(measures))
    width = 0.24
    fig, axis = plt.subplots(figsize=(8.8, 5.2))
    for index, (model, color) in enumerate(zip(models, [COLORS["brand_url"], COLORS["html"], COLORS["full"]])):
        values = [float(subset.loc[model, column]) for column, _ in measures]
        bars = axis.bar(x + (index - 1) * width, values, width, label=model_label(model), color=color)
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=8.5)
    axis.set_xticks(x, [label for _, label in measures])
    axis.set_ylim(0.65, 1.01)
    axis.set_ylabel("Score")
    axis.set_title("CatBoost performance across feature views")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.24))
    save(fig, output / "figure_4_1_feature_view_comparison.png", dpi)


def plot_representative_models(metrics: pd.DataFrame, output: Path, dpi: int) -> None:
    models = [
        "full__catboost",
        "full__xgboost",
        "full__lightgbm",
        "full__extra_trees",
        "full__stacking_boosting",
        "full__stacking_heterogeneous",
    ]
    measures = [("f1", "F1"), ("pr_auc", "PR-AUC"), ("mcc", "MCC")]
    subset = metrics.loc[models]
    x = np.arange(len(models))
    width = 0.25
    fig, axis = plt.subplots(figsize=(11.5, 5.8))
    colors = ["#377eb8", "#4daf4a", "#7b5ea7"]
    for index, ((column, label), color) in enumerate(zip(measures, colors)):
        values = subset[column].astype(float).to_numpy()
        bars = axis.bar(x + (index - 1) * width, values, width, label=label, color=color)
        axis.bar_label(
            bars,
            labels=[f"{value:.3f}" for value in values],
            padding=4,
            fontsize=8.2,
            rotation=0,
        )
    tick_labels = [
        "CatBoost",
        "XGBoost",
        "LightGBM",
        "Extra Trees",
        "Boosting\nStacking",
        "Heterogeneous\nStacking",
    ]
    axis.set_xticks(x, tick_labels, rotation=0, ha="center")
    axis.set_ylim(0.80, 1.025)
    axis.set_ylabel("Score")
    axis.set_title(
        "Representative single models and stacking ensembles on the Full view",
        pad=22,
    )
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.34))
    save(fig, output / "figure_4_2_single_vs_stacking.png", dpi)


def curve_models() -> list[tuple[str, str]]:
    return [
        ("brand_url__catboost", COLORS["brand_url"]),
        ("html__catboost", COLORS["html"]),
        ("full__catboost", COLORS["full"]),
        ("full__stacking_heterogeneous", COLORS["stack"]),
    ]


def plot_pr_roc(predictions: pd.DataFrame, output: Path, dpi: int) -> None:
    y = predictions["label_binary"].astype(int).to_numpy()
    line_styles = ["-", "--", "-.", ":"]

    # Precision–Recall is kept as a standalone figure so labels and the
    # separation between the strongest curves remain legible in a thesis page.
    fig_pr, axis_pr = plt.subplots(figsize=(8.8, 6.2))
    for (model, color), style in zip(curve_models(), line_styles):
        score = predictions[score_column(model)].astype(float).to_numpy()
        precision, recall, _ = precision_recall_curve(y, score)
        ap = average_precision_score(y, score)
        axis_pr.plot(
            recall,
            precision,
            color=color,
            linestyle=style,
            linewidth=2.2,
            label=f"{model_label(model)} (AP={ap:.3f})",
        )
    prevalence = float(y.mean())
    axis_pr.axhline(
        prevalence,
        color="#777777",
        linewidth=1.2,
        linestyle="--",
        label=f"Prevalence={prevalence:.3f}",
    )
    axis_pr.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision–Recall curves",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis_pr.grid(alpha=0.22)
    axis_pr.legend(frameon=False, fontsize=9, loc="lower left")
    save(fig_pr, output / "figure_4_3_precision_recall_curve.png", dpi)

    # ROC uses the same model/color/style mapping in a second standalone file.
    fig_roc, axis_roc = plt.subplots(figsize=(8.8, 6.2))
    for (model, color), style in zip(curve_models(), line_styles):
        score = predictions[score_column(model)].astype(float).to_numpy()
        fpr, tpr, _ = roc_curve(y, score)
        auc = roc_auc_score(y, score)
        axis_roc.plot(
            fpr,
            tpr,
            color=color,
            linestyle=style,
            linewidth=2.2,
            label=f"{model_label(model)} (AUC={auc:.3f})",
        )
    axis_roc.plot(
        [0, 1], [0, 1], color="#777777", linewidth=1.2,
        linestyle="--", label="Random classifier",
    )
    axis_roc.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="ROC curves",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis_roc.grid(alpha=0.22)
    axis_roc.legend(frameon=False, fontsize=9, loc="lower right")
    save(fig_roc, output / "figure_4_4_roc_curve.png", dpi)


def plot_confusion(predictions: pd.DataFrame, output: Path, threshold: float, dpi: int) -> None:
    model = "full__stacking_boosting"
    y = predictions["label_binary"].astype(int).to_numpy()
    score = predictions[score_column(model)].astype(float).to_numpy()
    prediction = (score >= threshold).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y, prediction, display_labels=["Legitimate", "Phishing"], cmap="Blues",
        colorbar=False, values_format="d", ax=axes[0],
    )
    axes[0].set_title("Counts")
    ConfusionMatrixDisplay.from_predictions(
        y, prediction, display_labels=["Legitimate", "Phishing"], cmap="Blues",
        colorbar=False, normalize="true", values_format=".2%", ax=axes[1],
    )
    axes[1].set_title("Normalized by true class")
    fig.suptitle(f"Full Boosting Stacking confusion matrix (threshold={threshold:.2f})", y=1.02)
    save(fig, output / "figure_4_6_confusion_matrix.png", dpi)


def plot_calibration(predictions: pd.DataFrame, metrics: pd.DataFrame, output: Path, dpi: int) -> None:
    models = ["full__catboost", "full__stacking_boosting", "full__stacking_heterogeneous"]
    y = predictions["label_binary"].astype(int).to_numpy()
    fig, axis = plt.subplots(figsize=(7.2, 6.0))
    for model, color, marker in zip(models, [COLORS["full"], "#377eb8", COLORS["stack"]], ["o", "s", "^"]):
        score = predictions[score_column(model)].astype(float).to_numpy()
        observed, predicted = calibration_curve(y, score, n_bins=15, strategy="quantile")
        brier = float(metrics.loc[model, "brier"])
        ece = float(metrics.loc[model, "ece_15"])
        axis.plot(predicted, observed, marker=marker, markersize=4, linewidth=1.8, color=color,
                  label=f"{model_label(model)} (Brier={brier:.3f}, ECE={ece:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1.2, label="Perfect calibration")
    axis.set(xlabel="Mean predicted probability", ylabel="Observed phishing rate",
             title="Probability calibration on Test", xlim=(0, 1), ylim=(0, 1))
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, fontsize=8.4, loc="upper left")
    save(fig, output / "figure_4_5_calibration_curve.png", dpi)


def export_tables(metrics: pd.DataFrame, output: Path) -> None:
    columns = [
        "model", "accuracy", "precision", "recall", "specificity", "f1",
        "balanced_accuracy", "pr_auc", "roc_auc", "mcc", "fpr", "fnr",
        "brier", "ece_15", "tn", "fp", "fn", "tp",
    ]
    metrics.reset_index(drop=True)[columns].sort_values(
        ["pr_auc", "mcc", "f1"], ascending=False
    ).to_csv(output / "table_4_1_all_test_models.csv", index=False, encoding="utf-8-sig")
    selected = [
        "brand_url__catboost", "html__catboost", "full__catboost",
        "full__stacking_boosting", "full__stacking_heterogeneous",
    ]
    summary = metrics.loc[selected, columns].copy()
    summary.insert(1, "display_name", [model_label(model) for model in selected])
    summary.to_csv(output / "table_4_2_representative_models.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    args = arguments()
    set_style()
    experiment = project_path(args.experiment_dir, must_exist=True)
    output = project_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(experiment / "predictions" / "test.csv", low_memory=False)
    metrics = test_metrics(pd.read_csv(experiment / "metrics.csv"))

    required_models = {model for model, _ in curve_models()} | {
        "full__stacking_boosting", "full__catboost", "full__xgboost",
        "full__lightgbm", "full__extra_trees",
    }
    missing_metrics = sorted(required_models - set(metrics.index))
    missing_scores = sorted(score_column(model) for model in required_models if score_column(model) not in predictions)
    if missing_metrics or missing_scores:
        raise ValueError(f"Missing metrics={missing_metrics}; missing score columns={missing_scores}")

    plot_feature_views(metrics, output, args.dpi)
    plot_representative_models(metrics, output, args.dpi)
    plot_pr_roc(predictions, output, args.dpi)
    plot_confusion(predictions, output, args.threshold, args.dpi)
    plot_calibration(predictions, metrics, output, args.dpi)
    export_tables(metrics, output)

    manifest = {
        "project_author": PROJECT_AUTHOR,
        "project_repository": PROJECT_REPOSITORY,
        "source_experiment": str(experiment),
        "test_rows": int(len(predictions)),
        "threshold": float(args.threshold),
        "model_training_performed": False,
        "figures": sorted(path.name for path in output.glob("figure_*.png")),
        "tables": sorted(path.name for path in output.glob("table_*.csv")),
    }
    (output / "chapter4_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Saved Chapter 4 figures: {output}")


if __name__ == "__main__":
    main()
