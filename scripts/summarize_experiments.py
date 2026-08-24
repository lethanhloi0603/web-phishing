import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import binary_metrics


METRICS = ("precision", "recall", "f1", "auprc", "auroc", "fpr", "fnr", "mcc", "brier", "ece")


def evaluations(report):
    values = {"fixed_0_5": (0.5, report["overall"])}
    for mode, calibration in report.get("calibration", {}).items():
        values[mode] = (float(calibration["threshold"]), calibration["test"])
    return values


def stratified_bootstrap_indices(
    predictions, rng, cluster_column=None, cluster_strata_column="site_label"
):
    """Sample pages while preserving sites and their page-label composition.

    A phishing site may legitimately contain benign auxiliary pages alongside
    credential-entry pages.  Clustered sampling therefore stratifies sites by
    ``site_label`` when that column is available, while metrics continue to use
    the page-level ``label`` column.
    """
    labels = predictions["label"].astype(int).to_numpy()
    if cluster_column is None:
        positive = np.flatnonzero(labels == 1)
        negative = np.flatnonzero(labels == 0)
        if not len(positive) or not len(negative):
            return np.asarray([], dtype=int)
        return np.concatenate([
            rng.choice(positive, size=len(positive), replace=True),
            rng.choice(negative, size=len(negative), replace=True),
        ])

    if cluster_column not in predictions:
        raise ValueError(
            f"Cluster bootstrap requires prediction column '{cluster_column}'"
        )
    clusters = predictions[cluster_column].fillna("").astype(str).copy()
    empty = clusters.str.strip().eq("")
    fallback = (
        predictions["sample_id"].astype(str)
        if "sample_id" in predictions
        else predictions.index.astype(str).to_series(index=predictions.index)
    )
    clusters.loc[empty] = fallback.loc[empty]
    strata = labels
    if cluster_strata_column and cluster_strata_column in predictions:
        candidate = pd.to_numeric(
            predictions[cluster_strata_column], errors="coerce"
        ).to_numpy()
        if np.isin(candidate[~pd.isna(candidate)], [0, 1]).all() and not pd.isna(candidate).any():
            strata = candidate.astype(int)

    groups = []
    for cluster, index in predictions.groupby(clusters, sort=False).groups.items():
        positions = predictions.index.get_indexer(index)
        group_strata = np.unique(strata[positions])
        if len(group_strata) != 1:
            raise ValueError(
                f"Cluster '{cluster}' has conflicting labels in its bootstrap stratum; "
                "clustered bootstrap is invalid"
            )
        groups.append((int(group_strata[0]), positions))
    by_label = {
        label: [positions for group_label, positions in groups if group_label == label]
        for label in (0, 1)
    }
    if not by_label[0] or not by_label[1]:
        return np.asarray([], dtype=int)
    sampled = []
    for label in (1, 0):
        group_indices = rng.integers(0, len(by_label[label]), size=len(by_label[label]))
        sampled.extend(by_label[label][int(index)] for index in group_indices)
    return np.concatenate(sampled).astype(int, copy=False)


def hierarchical_bootstrap(
    runs, evaluation_name, samples, rng, cluster_column="codebase_id"
):
    predictions = runs[0]["predictions"]
    labels = predictions["label"].astype(int).to_numpy()
    if not (labels == 1).any() or not (labels == 0).any():
        return {metric: (float("nan"), float("nan")) for metric in METRICS}

    draws = {metric: [] for metric in METRICS}
    seed_count = len(runs)
    for _ in range(samples):
        sampled_rows = stratified_bootstrap_indices(
            predictions, rng, cluster_column=cluster_column
        )
        sampled_seeds = rng.integers(0, seed_count, size=seed_count)
        per_seed = []
        for seed_index in sampled_seeds:
            run = runs[int(seed_index)]
            probability = run["predictions"]["probability"].astype(float).to_numpy()
            threshold = run["evaluations"][evaluation_name][0]
            per_seed.append(binary_metrics(
                labels[sampled_rows], probability[sampled_rows], threshold=threshold
            ))
        for metric in METRICS:
            draws[metric].append(float(np.mean([item[metric] for item in per_seed])))
    return {
        metric: (
            float(np.nanpercentile(values, 2.5)),
            float(np.nanpercentile(values, 97.5)),
        )
        for metric, values in draws.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=123)
    parser.add_argument(
        "--bootstrap-unit", choices=["site", "page"], default="site",
        help="Use site-clustered bootstrap by default to avoid page-level pseudo-replication",
    )
    parser.add_argument("--cluster-column", default="codebase_id")
    args = parser.parse_args()

    root = Path(args.root)
    grouped = {}
    for metrics_path in sorted(root.glob("*/seed_*/metrics.json")):
        variant = metrics_path.parent.parent.name
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        prediction_path = metrics_path.parent / "predictions.csv"
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing predictions for bootstrap: {prediction_path}")
        predictions = pd.read_csv(prediction_path)
        grouped.setdefault(variant, []).append({
            "seed": int(metrics_path.parent.name.replace("seed_", "")),
            "report": report,
            "evaluations": evaluations(report),
            "predictions": predictions,
        })
    if not grouped:
        raise RuntimeError(f"No */seed_*/metrics.json files found under {root}")

    rng = np.random.default_rng(args.bootstrap_seed)
    cluster_column = args.cluster_column if args.bootstrap_unit == "site" else None
    rows = []
    for variant, runs in sorted(grouped.items()):
        runs.sort(key=lambda item: item["seed"])
        reference = runs[0]["predictions"]
        for run in runs[1:]:
            if len(run["predictions"]) != len(reference):
                raise ValueError(f"Prediction row mismatch for variant {variant}")
            if "sample_id" in reference and not run["predictions"]["sample_id"].equals(reference["sample_id"]):
                raise ValueError(f"Prediction sample order mismatch for variant {variant}")

        common_evaluations = set.intersection(
            *(set(run["evaluations"]) for run in runs)
        )
        for evaluation_name in sorted(common_evaluations):
            bootstrap = hierarchical_bootstrap(
                runs, evaluation_name, args.bootstrap_samples, rng,
                cluster_column=cluster_column,
            )
            for metric in METRICS:
                values = np.asarray([
                    run["evaluations"][evaluation_name][1][metric] for run in runs
                ], dtype=float)
                mean = float(np.nanmean(values))
                std = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
                margin = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
                rows.append({
                    "variant": variant,
                    "evaluation": evaluation_name,
                    "metric": metric,
                    "n_seeds": len(values),
                    "mean": mean,
                    "std": std,
                    "normal_ci95_low": mean - margin,
                    "normal_ci95_high": mean + margin,
                    "bootstrap_ci95_low": bootstrap[metric][0],
                    "bootstrap_ci95_high": bootstrap[metric][1],
                    "bootstrap_unit": args.bootstrap_unit,
                    "cluster_column": cluster_column or "",
                    "seeds": ",".join(str(run["seed"]) for run in runs),
                })

    output_frame = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(out_csv, index=False)
    out_json.write_text(
        json.dumps(output_frame.to_dict("records"), indent=2), encoding="utf-8"
    )
    print(f"Summary CSV:  {out_csv}")
    print(f"Summary JSON: {out_json}")
    key = output_frame[
        output_frame["metric"].isin(["f1", "recall", "fpr", "auprc"])
    ]
    print(key[["variant", "evaluation", "metric", "mean", "std", "bootstrap_ci95_low", "bootstrap_ci95_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
