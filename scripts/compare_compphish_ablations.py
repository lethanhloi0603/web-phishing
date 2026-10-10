# Copyright (c) 2026 Le Thanh Loi. All rights reserved.
# Author: Le Thanh Loi

"""Paired comparison of CompPhish baseline and shortcut ablations.

Every experiment keeps its model and threshold selected from Train OOF.  The
same resampled Test indices are then used for all variants, so confidence
intervals describe *paired* changes rather than unrelated model uncertainty.
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
from scipy.stats import binomtest

from compphish_common import evaluate_binary, project_path


METRICS = [
    "accuracy", "precision", "recall", "specificity", "f1",
    "pr_auc", "roc_auc", "mcc", "fpr", "fnr",
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--drop-www-dir", required=True)
    parser.add_argument("--drop-visible-dir", required=True)
    parser.add_argument(
        "--fold-safe-70-dir",
        default=None,
        help="Optional 70-feature fold-safe experiment compared with baseline.",
    )
    parser.add_argument(
        "--output-dir", default="artifacts/compphish/ablation_comparison_integrity"
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_variant(name: str, directory: str) -> dict[str, object]:
    root = project_path(directory, must_exist=True)
    selected = json.loads((root / "selected_model.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(root / "predictions" / "test.csv")
    model = str(selected["model"])
    score_column = f"score__{model}"
    if score_column not in predictions.columns:
        raise ValueError(f"{name}: missing {score_column}")
    return {
        "name": name,
        "root": root,
        "model": model,
        "threshold": float(selected["threshold"]),
        "frame": predictions[
            ["sample_id", "label_binary", score_column]
        ].rename(columns={score_column: "score"}),
    }


def align(variants: list[dict[str, object]]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    reference = variants[0]["frame"].sort_values("sample_id").reset_index(drop=True)
    ids = reference["sample_id"].astype(str).to_numpy()
    y = reference["label_binary"].to_numpy(dtype=int)
    scores: dict[str, np.ndarray] = {}
    for variant in variants:
        frame = variant["frame"].sort_values("sample_id").reset_index(drop=True)
        if not np.array_equal(ids, frame["sample_id"].astype(str).to_numpy()):
            raise ValueError(f"Test sample IDs do not align for {variant['name']}")
        if not np.array_equal(y, frame["label_binary"].to_numpy(dtype=int)):
            raise ValueError(f"Test labels do not align for {variant['name']}")
        scores[str(variant["name"])] = frame["score"].to_numpy(dtype=float)
    return y, scores


def main() -> None:
    args = arguments()
    output = project_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    variants = [
        load_variant("baseline", args.baseline_dir),
        load_variant("drop_count_www", args.drop_www_dir),
        load_variant("drop_visible_url_cues", args.drop_visible_dir),
    ]
    if args.fold_safe_70_dir:
        variants.append(load_variant("fold_safe_70", args.fold_safe_70_dir))
    y, scores = align(variants)

    points: dict[str, dict[str, float | int]] = {}
    rows = []
    for variant in variants:
        name = str(variant["name"])
        result = evaluate_binary(y, scores[name], float(variant["threshold"]))
        points[name] = result
        rows.append({
            "variant": name,
            "selected_model": variant["model"],
            "train_oof_threshold": variant["threshold"],
            **result,
        })
    pd.DataFrame(rows).to_csv(
        output / "table_ablation_point_estimates.csv", index=False, encoding="utf-8-sig"
    )

    rng = np.random.default_rng(args.seed)
    comparison_names = [str(variant["name"]) for variant in variants[1:]]
    deltas = {
        comparison: {metric: [] for metric in METRICS}
        for comparison in comparison_names
    }
    skipped = 0
    for iteration in range(args.bootstrap_iterations):
        indices = rng.integers(0, len(y), len(y))
        if len(np.unique(y[indices])) < 2:
            skipped += 1
            continue
        baseline = evaluate_binary(
            y[indices], scores["baseline"][indices], float(variants[0]["threshold"])
        )
        for variant in variants[1:]:
            name = str(variant["name"])
            result = evaluate_binary(
                y[indices], scores[name][indices], float(variant["threshold"])
            )
            for metric in METRICS:
                deltas[name][metric].append(float(result[metric]) - float(baseline[metric]))
        if (iteration + 1) % 250 == 0:
            print(f"Paired bootstrap {iteration + 1}/{args.bootstrap_iterations}")

    delta_rows = []
    for comparison in comparison_names:
        for metric in METRICS:
            values = np.asarray(deltas[comparison][metric], dtype=float)
            point_delta = float(points[comparison][metric]) - float(points["baseline"][metric])
            delta_rows.append({
                "comparison": f"{comparison}_minus_baseline",
                "metric": metric,
                "point_delta": point_delta,
                "ci_95_lower": float(np.nanquantile(values, 0.025)),
                "ci_95_upper": float(np.nanquantile(values, 0.975)),
                "ci_excludes_zero": bool(
                    np.nanquantile(values, 0.025) > 0
                    or np.nanquantile(values, 0.975) < 0
                ),
                "bootstrap_iterations_used": int(np.sum(np.isfinite(values))),
            })
    pd.DataFrame(delta_rows).to_csv(
        output / "table_paired_bootstrap_deltas.csv", index=False, encoding="utf-8-sig"
    )

    baseline_prediction = scores["baseline"] >= float(variants[0]["threshold"])
    baseline_correct = baseline_prediction == y
    mcnemar_rows = []
    for variant in variants[1:]:
        name = str(variant["name"])
        prediction = scores[name] >= float(variant["threshold"])
        correct = prediction == y
        baseline_only = int(np.sum(baseline_correct & ~correct))
        ablation_only = int(np.sum(~baseline_correct & correct))
        discordant = baseline_only + ablation_only
        p_value = (
            float(binomtest(min(baseline_only, ablation_only), discordant, 0.5).pvalue)
            if discordant else 1.0
        )
        mcnemar_rows.append({
            "comparison": f"{name}_vs_baseline",
            "baseline_correct_ablation_wrong": baseline_only,
            "baseline_wrong_ablation_correct": ablation_only,
            "discordant_pairs": discordant,
            "exact_mcnemar_p_value": p_value,
        })
    pd.DataFrame(mcnemar_rows).to_csv(
        output / "table_mcnemar_ablations.csv", index=False, encoding="utf-8-sig"
    )

    plot = pd.DataFrame(rows).set_index("variant")[["f1", "pr_auc", "mcc"]]
    axis = plot.plot.bar(figsize=(9, 5), color=["#2f6690", "#4c956c", "#c73e1d"])
    axis.set_ylim(max(0.0, float(plot.min().min()) - 0.08), 1.0)
    axis.set_xlabel("")
    axis.set_ylabel("Score on Test")
    axis.set_title("Shortcut ablation under the integrity-clean split")
    axis.tick_params(axis="x", rotation=0)
    axis.legend(loc="lower left")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "figure_ablation_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(axis.figure)

    summary = {
        "test_rows": len(y),
        "variants": {
            str(variant["name"]): {
                "selected_model": variant["model"],
                "threshold_selected_on_train_oof": variant["threshold"],
                "test_metrics": points[str(variant["name"])],
            }
            for variant in variants
        },
        "paired_bootstrap_iterations_requested": args.bootstrap_iterations,
        "paired_bootstrap_replicates_skipped": skipped,
        "test_used_for_selection": False,
    }
    (output / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(pd.DataFrame(rows)[
        ["variant", "selected_model", "accuracy", "precision", "recall", "f1", "pr_auc", "mcc"]
    ].to_string(index=False))
    print(f"Saved paired ablation analysis: {output}")


if __name__ == "__main__":
    main()
