# Copyright (c) 2026 Le Thanh Loi. All rights reserved.
# Author: Le Thanh Loi

"""Build descriptive tables and figures from a completed CompPhish run.

The thesis protocol intentionally excludes paired bootstrap and McNemar tests.
This script therefore reports point estimates only and does not make
statistical-significance claims.
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd
from compphish_common import evaluate_binary, project_path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir", default="artifacts/compphish/branch_stacking_full"
    )
    parser.add_argument(
        "--output-dir", default="artifacts/compphish/branch_stacking_analysis"
    )
    parser.add_argument(
        "--bootstrap-iterations", type=int, default=0,
        help="Deprecated and ignored; retained only for command compatibility.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def model_family(name: str) -> str:
    if "__stacking_" in name:
        return f"{name.split('__', 1)[0]}_stacking"
    if name in {"hard_voting", "soft_voting", "weighted_soft_voting", "stacking_logreg"}:
        return "ensemble"
    if name.startswith("brand__"):
        return "brand"
    if name.startswith("url__"):
        return "url"
    if name.startswith("brand_url__"):
        return "brand_url"
    if name.startswith("html__"):
        return "html"
    if name.startswith("full__"):
        return "full"
    if name.startswith("url_text__"):
        return "url_text"
    return "other"


def main() -> None:
    args = arguments()
    experiment = project_path(args.experiment_dir, must_exist=True)
    output = project_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(experiment / "metrics.csv")
    predictions_path = experiment / "predictions" / "test.csv"
    if not predictions_path.exists():
        # Compatibility with experiments produced by the earlier pipeline.
        predictions_path = experiment / "predictions" / "test_locked.csv"
    predictions = pd.read_csv(predictions_path)
    selected = json.loads((experiment / "selected_model.json").read_text(encoding="utf-8"))
    selected_name = selected["model"]
    threshold = float(selected["threshold"])
    score_column = f"score__{selected_name}"
    if score_column not in predictions:
        raise ValueError(f"Missing selected score column: {score_column}")

    test_metrics = metrics[metrics["split"].isin(["test", "test_locked"])].copy()
    test_metrics["family"] = test_metrics["model"].map(model_family)
    test_metrics = test_metrics.sort_values(["pr_auc", "mcc", "f1"], ascending=False)
    test_metrics.to_csv(output / "table_all_test_models.csv", index=False, encoding="utf-8-sig")

    selection_metrics = metrics[
        metrics["split"].isin(["train_oof", "validation"])
    ].copy()
    selection_metrics["family"] = selection_metrics["model"].map(model_family)
    family_winners = (
        selection_metrics.sort_values(["pr_auc", "mcc", "f1"], ascending=False)
        .groupby("family", as_index=False)
        .first()[["family", "model"]]
    )
    family_table = family_winners.merge(
        test_metrics.drop(columns="family"), on="model", how="left"
    ).sort_values("pr_auc", ascending=False)
    family_table.to_csv(
        output / "table_best_model_by_feature_view.csv", index=False, encoding="utf-8-sig"
    )

    y = predictions["label_binary"].to_numpy(dtype=int)
    score = predictions[score_column].to_numpy(dtype=float)
    point = evaluate_binary(y, score, threshold)
    pd.DataFrame([{
        "model": selected_name,
        "decision_threshold": threshold,
        **point,
    }]).to_csv(
        output / "table_reference_model_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot = test_metrics.head(12).sort_values("pr_auc")
    fig, axis = plt.subplots(figsize=(10, max(5, 0.42 * len(plot))))
    axis.barh(plot["model"], plot["pr_auc"], color="#2f6690")
    axis.set_xlim(max(0.0, float(plot["pr_auc"].min()) - 0.03), 1.0)
    axis.set_xlabel("PR-AUC on Test")
    axis.set_ylabel("")
    axis.set_title("CompPhish model comparison")
    fig.tight_layout()
    fig.savefig(output / "figure_test_pr_auc.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    family_plot = family_table.sort_values("mcc")
    fig, axis = plt.subplots(figsize=(9, max(4, 0.55 * len(family_plot))))
    axis.barh(family_plot["family"], family_plot["mcc"], color="#c73e1d")
    axis.set_xlim(max(-0.05, float(family_plot["mcc"].min()) - 0.05), 1.0)
    axis.set_xlabel("MCC on Test")
    axis.set_ylabel("Feature view / ensemble")
    axis.set_title("Best Train-OOF-selected model from each feature view")
    fig.tight_layout()
    fig.savefig(output / "figure_feature_view_mcc.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "selected_model": selected_name,
        "decision_threshold": threshold,
        "test_rows": len(y),
        "test_metrics": point,
        "inference_tests": "not used (descriptive comparison protocol)",
        "test_used_for_selection": False,
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved final analysis: {output}")


if __name__ == "__main__":
    main()
