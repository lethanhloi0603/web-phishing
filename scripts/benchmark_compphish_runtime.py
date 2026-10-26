# Copyright (c) 2026 Le Thanh Loi. All rights reserved.
# Author: Le Thanh Loi

"""Benchmark training/inference cost for the CompPhish Full-view models.

This script is deliberately separate from the main experiment.  It does not
select models or overwrite research metrics.  Single-model training time is a
single fit on all Train rows.  Stacking training time includes five-fold OOF
generation for every base member, refitting those members on all Train rows,
and fitting the logistic-regression meta learner.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
from pathlib import Path
from statistics import median
from time import perf_counter
from types import SimpleNamespace

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from compphish_common import RANDOM_STATE, project_path, read_processed
from train_compphish_models import STACKING_RECIPES, model_specs, scores


COLORS = {
    "single": "#377eb8",
    "stacking": "#e6550d",
    "catboost": "#31a354",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/processed/compphish/compphish_integrity_split.csv.gz",
    )
    parser.add_argument(
        "--metrics",
        default="artifacts/compphish/branch_stacking_full/metrics.csv",
        help="Existing final metrics; read only for the trade-off figure.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/compphish/runtime_benchmark_full",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--inference-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate figures from runtime_summary.csv without fitting models.",
    )
    return parser.parse_args()


def training_arguments(args: argparse.Namespace) -> SimpleNamespace:
    """Construct the exact Full-view framework configuration."""
    return SimpleNamespace(
        seed=args.seed,
        quick=False,
        model_set="framework",
        extended_all_views=False,
        views=["full"],
        feature_selector="none",
        selector_k=0,
        optional_boosters="required",
        max_tfidf_features=100000,
        include_fold_safe_mean_features=True,
        drop_features=[],
        drop_visible_url_cues=False,
    )


def short_label(name: str) -> str:
    labels = {
        "logistic_regression": "Logistic Regression",
        "gaussian_naive_bayes": "Gaussian NB",
        "knn": "KNN",
        "decision_tree": "Decision Tree",
        "rbf_svm": "RBF-SVM",
        "mlp": "MLP",
        "random_forest": "Random Forest",
        "extra_trees": "Extra Trees",
        "hist_gradient_boosting": "HistGradientBoosting",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "catboost": "CatBoost",
        "stacking_boosting": "Boosting Stacking",
        "stacking_heterogeneous": "Heterogeneous Stacking",
    }
    return labels.get(name.removeprefix("full__"), name)


def timed_scores(model: object, values: pd.DataFrame, repeats: int) -> tuple[float, np.ndarray]:
    # One unmeasured warm-up reduces one-time dispatch/cache effects.
    result = scores(model, values)
    durations = []
    for _ in range(repeats):
        started = perf_counter()
        result = scores(model, values)
        durations.append(perf_counter() - started)
    return median(durations), result


def serialized_size_mb(model: object, path: Path) -> float:
    joblib.dump(model, path, compress=3)
    size = path.stat().st_size / (1024 * 1024)
    path.unlink()
    return size


def make_summary(raw: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "training_seconds",
        "inference_seconds_test",
        "inference_ms_per_sample",
        "serialized_size_mb",
    ]
    rows = []
    for (model, kind, members), group in raw.groupby(
        ["model", "model_kind", "component_count"], sort=False
    ):
        row = {
            "model": model,
            "model_label": short_label(model),
            "model_kind": kind,
            "component_count": members,
            "repeats": len(group),
        }
        for column in numeric:
            values = group[column].astype(float)
            row[f"median_{column}"] = float(values.median())
            row[f"min_{column}"] = float(values.min())
            row[f"max_{column}"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("median_training_seconds")


def add_metrics(summary: pd.DataFrame, metrics_path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_path)
    test = metrics[metrics["split"].eq("test")][
        ["model", "accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc", "mcc"]
    ]
    return summary.merge(test, on="model", how="left", validate="one_to_one")


def plot_training(summary: pd.DataFrame, output: Path, dpi: int) -> None:
    ordered = summary.sort_values("median_training_seconds", ascending=False)
    colors = [
        COLORS["stacking"] if kind == "stacking" else COLORS["single"]
        for kind in ordered["model_kind"]
    ]
    fig, ax = plt.subplots(figsize=(10.8, 7.2))
    values = ordered["median_training_seconds"].to_numpy(dtype=float)
    bars = ax.barh(
        ordered["model_label"], values, color=colors,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Thời gian huấn luyện trung vị (giây)")
    ax.set_title(
        "Thời gian huấn luyện của các mô hình trên nhóm đặc trưng Full",
        pad=18,
        weight="bold",
    )
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    upper = float(values.max())
    ax.set_xlim(0, upper * 1.16)
    for bar, value in zip(bars, values):
        ax.text(
            value + upper * 0.012, bar.get_y() + bar.get_height() / 2,
            f"{value:.2f} giây", va="center", fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(summary: pd.DataFrame, output: Path, dpi: int) -> None:
    usable = summary.dropna(subset=["f1", "median_training_seconds"]).copy()
    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    label_offsets = {
        "full__catboost": (6, 13),
        "full__xgboost": (-26, 14),
        "full__lightgbm": (7, -10),
        "full__extra_trees": (7, 8),
        "full__stacking_boosting": (7, 8),
        "full__stacking_heterogeneous": (7, 8),
    }
    for _, row in usable.iterrows():
        color = COLORS["stacking"] if row["model_kind"] == "stacking" else COLORS["single"]
        marker = "D" if row["model_kind"] == "stacking" else "o"
        size = 100 if row["model_kind"] == "stacking" else 55
        if row["model"] == "full__catboost":
            color, marker, size = COLORS["catboost"], "s", 90
        ax.scatter(
            row["median_training_seconds"], row["f1"],
            color=color, marker=marker, s=size, alpha=0.9,
            edgecolor="white", linewidth=0.7,
        )
        if row["model_kind"] == "stacking" or row["model"] in {
            "full__catboost", "full__xgboost", "full__lightgbm",
            "full__extra_trees",
        }:
            ax.annotate(
                row["model_label"],
                (row["median_training_seconds"], row["f1"]),
                xytext=label_offsets.get(row["model"], (6, 6)),
                textcoords="offset points", fontsize=8,
            )
    ax.set_xlabel("Thời gian huấn luyện trung vị (giây)")
    ax.set_ylabel("F1-score trên tập Test")
    ax.set_title(
        "Đánh đổi giữa F1-score và thời gian huấn luyện trên nhóm Full",
        pad=18,
        weight="bold",
    )
    ax.set_xlim(left=0)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_inference(summary: pd.DataFrame, output: Path, dpi: int) -> None:
    ordered = summary.sort_values(
        "median_inference_ms_per_sample", ascending=False
    )
    colors = [
        COLORS["stacking"] if kind == "stacking" else COLORS["single"]
        for kind in ordered["model_kind"]
    ]
    fig, ax = plt.subplots(figsize=(10.8, 7.2))
    values = ordered["median_inference_ms_per_sample"].to_numpy(dtype=float)
    bars = ax.barh(
        ordered["model_label"], values, color=colors,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Thời gian suy luận trung vị (ms/mẫu)")
    ax.set_title(
        "Thời gian suy luận của các mô hình trên nhóm đặc trưng Full",
        pad=18,
        weight="bold",
    )
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    upper = float(values.max())
    ax.set_xlim(0, upper * 1.16)
    for bar, value in zip(bars, values):
        ax.text(
            value + upper * 0.012, bar.get_y() + bar.get_height() / 2,
            f"{value:.4f} ms", va="center", fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = arguments()
    if args.repeats < 1 or args.inference_repeats < 1:
        raise ValueError("--repeats and --inference-repeats must be positive")

    output = project_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        summary_path = output / "runtime_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = pd.read_csv(summary_path)
        plot_training(summary, output / "figure_runtime_training.png", args.dpi)
        plot_inference(summary, output / "figure_runtime_inference.png", args.dpi)
        plot_tradeoff(summary, output / "figure_runtime_f1_tradeoff.png", args.dpi)
        print(f"Regenerated runtime figures from: {summary_path}")
        return
    temp_model = output / "_runtime_size_probe.joblib"

    frame = read_processed(args.data)
    train = frame[~frame["split"].eq("test")].reset_index(drop=True)
    test = frame[frame["split"].eq("test")].reset_index(drop=True)
    y_train = train["label_binary"].to_numpy(dtype=int)
    groups = train["registered_domain"].to_numpy()
    folds = list(StratifiedGroupKFold(
        n_splits=args.cv_folds, shuffle=True, random_state=args.seed,
    ).split(train, y_train, groups=groups))

    specs = model_specs(training_arguments(args))
    if len(specs) != 12 or any(spec.view != "full" for spec in specs):
        raise RuntimeError(
            f"Expected 12 Full-view framework models, received {len(specs)}"
        )
    spec_by_name = {spec.name: spec for spec in specs}
    recipe_names = {
        recipe: [f"full__{member}" for member in members]
        for recipe, members in STACKING_RECIPES.items()
    }
    stack_members = set().union(*[set(value) for value in recipe_names.values()])

    rows: list[dict[str, object]] = []
    for repeat in range(1, args.repeats + 1):
        print(f"\n=== Timing repeat {repeat}/{args.repeats} ===", flush=True)
        fitted: dict[str, object] = {}
        full_fit_seconds: dict[str, float] = {}
        inference_seconds: dict[str, float] = {}
        oof_scores: dict[str, np.ndarray] = {}
        oof_seconds: dict[str, float] = {}

        rng = np.random.default_rng(args.seed + repeat)
        order = list(specs)
        rng.shuffle(order)
        for number, spec in enumerate(order, start=1):
            print(f"[{number}/{len(order)}] {spec.name}", flush=True)
            X_train = train[list(spec.columns)]
            X_test = test[list(spec.columns)]
            model = clone(spec.estimator)
            started = perf_counter()
            model.fit(X_train, y_train)
            full_fit_seconds[spec.name] = perf_counter() - started
            inference_seconds[spec.name], _ = timed_scores(
                model, X_test, args.inference_repeats
            )
            size_mb = serialized_size_mb(model, temp_model)
            fitted[spec.name] = model

            rows.append({
                "repeat": repeat,
                "model": spec.name,
                "model_kind": "single",
                "component_count": 1,
                "training_seconds": full_fit_seconds[spec.name],
                "inference_seconds_test": inference_seconds[spec.name],
                "inference_ms_per_sample": (
                    inference_seconds[spec.name] * 1000 / len(test)
                ),
                "serialized_size_mb": size_mb,
                "train_rows": len(train),
                "test_rows": len(test),
            })

            if spec.name in stack_members:
                member_oof = np.zeros(len(train), dtype=float)
                started = perf_counter()
                for fit_index, holdout_index in folds:
                    fold_model = clone(spec.estimator)
                    fold_model.fit(
                        train.iloc[fit_index][list(spec.columns)],
                        y_train[fit_index],
                    )
                    member_oof[holdout_index] = scores(
                        fold_model,
                        train.iloc[holdout_index][list(spec.columns)],
                    )
                oof_seconds[spec.name] = perf_counter() - started
                oof_scores[spec.name] = member_oof

        for recipe, members in recipe_names.items():
            stack_name = f"full__{recipe}"
            print(f"Timing {stack_name}", flush=True)
            Z_train = pd.DataFrame({name: oof_scores[name] for name in members})
            meta = LogisticRegression(max_iter=4000, random_state=args.seed)
            started = perf_counter()
            meta.fit(Z_train, y_train)
            meta_fit_seconds = perf_counter() - started

            def stack_predict() -> np.ndarray:
                Z_test = pd.DataFrame({
                    name: scores(
                        fitted[name],
                        test[list(spec_by_name[name].columns)],
                    )
                    for name in members
                })
                return meta.predict_proba(Z_test)[:, 1]

            stack_predict()  # warm-up
            durations = []
            for _ in range(args.inference_repeats):
                started = perf_counter()
                stack_predict()
                durations.append(perf_counter() - started)
            stack_inference = median(durations)

            stack_training = sum(
                full_fit_seconds[name] + oof_seconds[name] for name in members
            ) + meta_fit_seconds
            size_mb = sum(
                serialized_size_mb(fitted[name], temp_model) for name in members
            ) + serialized_size_mb(meta, temp_model)
            rows.append({
                "repeat": repeat,
                "model": stack_name,
                "model_kind": "stacking",
                "component_count": len(members) + 1,
                "training_seconds": stack_training,
                "inference_seconds_test": stack_inference,
                "inference_ms_per_sample": stack_inference * 1000 / len(test),
                "serialized_size_mb": size_mb,
                "train_rows": len(train),
                "test_rows": len(test),
            })

        pd.DataFrame(rows).to_csv(
            output / "runtime_raw.csv", index=False, encoding="utf-8-sig"
        )
        del fitted, oof_scores
        gc.collect()

    raw = pd.DataFrame(rows)
    summary = make_summary(raw)
    metrics_path = project_path(args.metrics, must_exist=True)
    summary = add_metrics(summary, metrics_path)
    summary.to_csv(
        output / "runtime_summary.csv", index=False, encoding="utf-8-sig"
    )

    plot_training(summary, output / "figure_runtime_training.png", args.dpi)
    plot_inference(summary, output / "figure_runtime_inference.png", args.dpi)
    plot_tradeoff(summary, output / "figure_runtime_f1_tradeoff.png", args.dpi)

    environment = {
        "benchmark_scope": "12 Full-view base learners and 2 Full-view stackers",
        "single_training_definition": "one fit on all Train rows",
        "stacking_training_definition": (
            "five-fold OOF fits plus full-Train refits of all members plus meta fit"
        ),
        "repeats": args.repeats,
        "inference_repeats_per_timing_repeat": args.inference_repeats,
        "train_rows": len(train),
        "test_rows": len(test),
        "cv_folds": args.cv_folds,
        "seed": args.seed,
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "parallelism": "native model settings from the final experiment",
        "metrics_source": str(metrics_path),
        "test_used_for_timing_only": True,
    }
    (output / "benchmark_environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    display = summary[[
        "model_label", "model_kind", "component_count",
        "median_training_seconds", "median_inference_ms_per_sample",
        "median_serialized_size_mb", "f1", "pr_auc", "mcc",
    ]]
    print("\nRuntime and predictive-performance summary:")
    print(display.to_string(index=False))
    print(f"\nSaved runtime benchmark: {output}")


if __name__ == "__main__":
    main()
