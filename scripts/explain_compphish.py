# Copyright (c) 2026 Le Thanh Loi. All rights reserved.
# Author: Le Thanh Loi

"""Generate two-level XAI outputs for the CompPhish experiment.

Level 1 explains how each branch-aligned OOF stacking layer combines its base
learners. Level 2 uses TreeSHAP and permutation importance to explain the
pre-specified full-feature CatBoost reference model.  Feature-level values are
not misreported as explanations of an entire heterogeneous ensemble.
"""

from __future__ import annotations

import argparse
import json

import joblib
import matplotlib

# This script only writes figures to disk.  Force a non-interactive backend so
# Windows/Tk objects are never created (joblib workers cannot safely destroy
# Tk objects outside the main thread).
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from compphish_common import (
    FEATURE_SETS,
    FEATURE_SETS_PUBLISHED,
    PROJECT_AUTHOR,
    PROJECT_REPOSITORY,
    project_path,
    read_processed,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/processed/compphish/compphish_integrity_split.csv.gz",
    )
    parser.add_argument(
        "--experiment-dir", default="artifacts/compphish/branch_stacking_full"
    )
    parser.add_argument(
        "--output-dir", default="artifacts/compphish/branch_stacking_xai_70"
    )
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--permutation-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--feature-model",
        default="full__catboost",
        help="Fitted tree model used for feature-level SHAP/permutation XAI.",
    )
    return parser.parse_args()


def positive_class_shap(values: object, feature_count: int) -> np.ndarray:
    """Normalize SHAP's version-dependent binary-class output shape."""
    if isinstance(values, list):
        array = np.asarray(values[-1])
    else:
        array = np.asarray(values)
    if array.ndim == 3:
        if array.shape[-1] == 2:
            array = array[:, :, 1]
        elif array.shape[0] == 2:
            array = array[1]
    if array.ndim != 2 or array.shape[1] != feature_count:
        raise ValueError(f"Unexpected SHAP shape: {array.shape}")
    return array


def feature_group(feature: str) -> str:
    if feature in FEATURE_SETS_PUBLISHED["brand_url"]:
        return "brand_url"
    return "html_content"


def main() -> None:
    args = arguments()
    experiment = project_path(args.experiment_dir, must_exist=True)
    output = project_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    try:
        import shap
    except ImportError as error:
        raise SystemExit("Install SHAP with: python -m pip install shap") from error

    frame = read_processed(args.data)
    test = frame[frame["split"].eq("test")].reset_index(drop=True)
    if test.empty:
        raise ValueError("The processed dataset contains no Test rows")

    branch_path = experiment / "models" / f"{args.feature_model}.joblib"
    if not branch_path.exists():
        raise FileNotFoundError(
            f"Missing {branch_path}. Run train_compphish_models.py first."
        )
    branch = joblib.load(branch_path)
    members_path = experiment / "base_members.csv"
    feature_columns = FEATURE_SETS["full"]
    if members_path.exists():
        members = pd.read_csv(members_path, dtype=str).fillna("")
        match = members[members["model"].eq(args.feature_model)]
        if not match.empty and match.iloc[0]["input_features"]:
            feature_columns = match.iloc[0]["input_features"].split("|")

    weights_path = experiment / "stacking_members.csv"
    if weights_path.exists():
        weights = pd.read_csv(weights_path)
        weights["abs_meta_coefficient"] = weights["meta_coefficient"].abs()
        weights.sort_values("abs_meta_coefficient", ascending=False).to_csv(
            output / "stacking_level_importance.csv", index=False, encoding="utf-8-sig"
        )
        for stack_name, stack_rows in weights.groupby("stacking_model"):
            plot_weights = stack_rows.sort_values("abs_meta_coefficient").tail(
                args.top_k
            )
            colors = np.where(
                plot_weights["meta_coefficient"] >= 0, "#c73e1d", "#2f6690"
            )
            fig, axis = plt.subplots(
                figsize=(9, max(4, 0.55 * len(plot_weights)))
            )
            axis.barh(
                plot_weights["base_member"],
                plot_weights["meta_coefficient"],
                color=colors,
            )
            axis.axvline(0, color="black", linewidth=0.8)
            axis.set_xlabel("Logistic stacking coefficient")
            axis.set_ylabel("")
            axis.set_title(f"Base-learner contribution: {stack_name}")
            fig.tight_layout()
            safe_name = stack_name.replace("__", "_")
            fig.savefig(
                output / f"figure_{safe_name}_coefficients.png",
                dpi=220,
                bbox_inches="tight",
            )
            plt.close(fig)

    sample_n = min(args.sample_size, len(test))
    sample = test.sample(n=sample_n, random_state=args.seed).reset_index(drop=True)
    transformed = branch[:-1].transform(sample[feature_columns])
    selector = branch.named_steps.get("selector")
    explained_features = feature_columns
    if selector is not None and selector != "passthrough" and hasattr(selector, "get_support"):
        explained_features = [
            feature for feature, keep in zip(feature_columns, selector.get_support()) if keep
        ]
    estimator = branch.named_steps["model"]
    explainer = shap.TreeExplainer(estimator)
    shap_values = positive_class_shap(
        explainer.shap_values(transformed), len(explained_features)
    )

    global_importance = pd.DataFrame({
        "feature": explained_features,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "mean_signed_shap": shap_values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    global_importance["feature_group"] = global_importance["feature"].map(feature_group)
    global_importance.to_csv(
        output / "shap_global_importance.csv", index=False, encoding="utf-8-sig"
    )

    group_importance = global_importance.groupby("feature_group", as_index=False).agg(
        total_mean_abs_shap=("mean_abs_shap", "sum"),
        mean_feature_importance=("mean_abs_shap", "mean"),
        feature_count=("feature", "size"),
    ).sort_values("total_mean_abs_shap", ascending=False)
    group_importance["normalized_share"] = (
        group_importance["total_mean_abs_shap"]
        / group_importance["total_mean_abs_shap"].sum()
    )
    group_importance.to_csv(
        output / "shap_group_importance.csv", index=False, encoding="utf-8-sig"
    )

    top = global_importance.head(args.top_k).sort_values("mean_abs_shap")
    fig, axis = plt.subplots(figsize=(9, max(5, 0.34 * len(top))))
    axis.barh(top["feature"], top["mean_abs_shap"], color="#2f6690")
    axis.set_xlabel("Mean |SHAP value|")
    axis.set_ylabel("")
    axis.set_title(f"Global feature importance: {args.feature_model}")
    fig.tight_layout()
    fig.savefig(output / "figure_shap_global_bar.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    shap.summary_plot(
        shap_values,
        transformed,
        feature_names=explained_features,
        max_display=args.top_k,
        show=False,
    )
    plt.title("SHAP distribution on a Test sample")
    plt.tight_layout()
    plt.savefig(output / "figure_shap_beeswarm.png", dpi=220, bbox_inches="tight")
    plt.close()

    print("Computing permutation importance on Test...")
    permutation = permutation_importance(
        branch,
        test[feature_columns],
        test["label_binary"],
        scoring="average_precision",
        n_repeats=args.permutation_repeats,
        random_state=args.seed,
        # A fitted estimator may itself use native worker threads.  Keeping the
        # outer permutation loop single-process avoids Tk/joblib teardown errors
        # on Windows and prevents nested parallelism from exhausting memory.
        n_jobs=1,
    )
    permutation_table = pd.DataFrame({
        "feature": feature_columns,
        "importance_mean": permutation.importances_mean,
        "importance_std": permutation.importances_std,
    }).sort_values("importance_mean", ascending=False)
    permutation_table["feature_group"] = permutation_table["feature"].map(feature_group)
    permutation_table.to_csv(
        output / "permutation_importance_test.csv", index=False, encoding="utf-8-sig"
    )

    branch_score = branch.predict_proba(test[feature_columns])[:, 1]
    branch_prediction = (branch_score >= 0.5).astype(int)
    categories = {
        "true_positive": (test["label_binary"].to_numpy() == 1) & (branch_prediction == 1),
        "true_negative": (test["label_binary"].to_numpy() == 0) & (branch_prediction == 0),
        "false_positive": (test["label_binary"].to_numpy() == 0) & (branch_prediction == 1),
        "false_negative": (test["label_binary"].to_numpy() == 1) & (branch_prediction == 0),
    }
    local_rows = []
    test_position_by_id = {value: index for index, value in enumerate(test["sample_id"])}
    sample_position_by_id = {value: index for index, value in enumerate(sample["sample_id"])}
    for category, mask in categories.items():
        candidate_ids = test.loc[mask, "sample_id"]
        candidate_ids = candidate_ids[candidate_ids.isin(sample_position_by_id)]
        if candidate_ids.empty:
            continue
        sample_id = candidate_ids.iloc[0]
        sample_position = sample_position_by_id[sample_id]
        test_position = test_position_by_id[sample_id]
        contribution = shap_values[sample_position]
        ranking = np.argsort(np.abs(contribution))[::-1][:10]
        for rank, feature_index in enumerate(ranking, start=1):
            local_rows.append({
                "case_type": category,
                "sample_id": sample_id,
                "url": test.iloc[test_position]["url"],
                "true_label": int(test.iloc[test_position]["label_binary"]),
                "branch_score": float(branch_score[test_position]),
                "rank": rank,
                "feature": explained_features[feature_index],
                "feature_value": float(transformed[sample_position, feature_index]),
                "shap_value": float(contribution[feature_index]),
            })
    pd.DataFrame(local_rows).to_csv(
        output / "local_case_explanations.csv", index=False, encoding="utf-8-sig"
    )

    scope = {
        "project_author": PROJECT_AUTHOR,
        "project_repository": PROJECT_REPOSITORY,
        "framework_feature_count": len(FEATURE_SETS_PUBLISHED["full"]),
        "ensemble_level": (
            "Exact logistic meta-classifier coefficients reported separately "
            "for every branch-aligned OOF stacking model."
        ),
        "feature_level": (
            f"TreeSHAP and permutation importance for {args.feature_model} only."
        ),
        "feature_model": args.feature_model,
        "warning": (
            f"Feature-level SHAP values explain {args.feature_model}, not a complete "
            "stacking ensemble. Stacking is explained separately at meta-model level."
        ),
        "shap_rows": sample_n,
        "test_rows_for_permutation": len(test),
        "feature_count": len(explained_features),
        "seed": args.seed,
    }
    (output / "xai_scope.json").write_text(
        json.dumps(scope, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(global_importance.head(args.top_k).to_string(index=False))
    print(f"Saved XAI outputs: {output}")


if __name__ == "__main__":
    main()
