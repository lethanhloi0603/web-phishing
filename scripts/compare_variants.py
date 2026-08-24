import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_experiments import stratified_bootstrap_indices


METRICS = {
    "f1": lambda y, p, t: f1_score(y, p >= t, zero_division=0),
    "mcc": lambda y, p, t: matthews_corrcoef(y, p >= t),
    "auprc": lambda y, p, _t: average_precision_score(y, p),
    "auroc": lambda y, p, _t: roc_auc_score(y, p),
}


def load_variant(root, variant, evaluation):
    runs = []
    for path in sorted((root / variant).glob("seed_*/predictions.csv")):
        metrics_path = path.with_name("metrics.json")
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        threshold = 0.5
        if evaluation != "fixed_0_5":
            threshold = float(report["calibration"][evaluation]["threshold"])
        frame = pd.read_csv(path)
        runs.append((path.parent.name, threshold, frame))
    if not runs:
        raise FileNotFoundError(f"No runs found for variant '{variant}'")
    return runs


def align(left, right, cluster_column="codebase_id"):
    left_by_seed = {seed: (threshold, frame) for seed, threshold, frame in left}
    right_by_seed = {seed: (threshold, frame) for seed, threshold, frame in right}
    seeds = sorted(set(left_by_seed) & set(right_by_seed))
    if not seeds:
        raise ValueError("Variants have no common seeds")
    aligned = []
    for seed in seeds:
        lt, lf = left_by_seed[seed]
        rt, rf = right_by_seed[seed]
        key = "sample_id" if "sample_id" in lf and "sample_id" in rf else None
        if key:
            if cluster_column and (cluster_column not in lf or cluster_column not in rf):
                raise ValueError(
                    f"Clustered comparison requires prediction column '{cluster_column}'"
                )
            left_columns = [key, "label", "probability"]
            right_columns = [key, "label", "probability"]
            if cluster_column:
                left_columns.append(cluster_column)
                right_columns.append(cluster_column)
            merged = lf[left_columns].merge(
                rf[right_columns], on=key, suffixes=("_a", "_b"), validate="one_to_one"
            )
            if not merged["label_a"].equals(merged["label_b"]):
                raise ValueError(f"Label mismatch for {seed}")
            y = merged["label_a"].astype(int).to_numpy()
            pa = merged["probability_a"].astype(float).to_numpy()
            pb = merged["probability_b"].astype(float).to_numpy()
            clusters = None
            if cluster_column:
                left_cluster = merged[f"{cluster_column}_a"].fillna("").astype(str)
                right_cluster = merged[f"{cluster_column}_b"].fillna("").astype(str)
                if not left_cluster.equals(right_cluster):
                    raise ValueError(f"Cluster mismatch for {seed}")
                clusters = left_cluster.to_numpy()
        else:
            if len(lf) != len(rf) or not lf["label"].equals(rf["label"]):
                raise ValueError(f"Prediction mismatch for {seed}")
            y = lf["label"].astype(int).to_numpy()
            pa = lf["probability"].astype(float).to_numpy()
            pb = rf["probability"].astype(float).to_numpy()
            if cluster_column and cluster_column not in lf:
                raise ValueError(
                    f"Clustered comparison requires prediction column '{cluster_column}'"
                )
            clusters = lf[cluster_column].fillna("").astype(str).to_numpy() if cluster_column else None
        aligned.append((seed, y, pa, pb, lt, rt, clusters))
    return aligned


def exact_mcnemar(y, pa, pb, ta, tb):
    correct_a = (pa >= ta) == y
    correct_b = (pb >= tb) == y
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    n = b + c
    if n == 0:
        return {"a_only_correct": b, "b_only_correct": c, "p_value": 1.0}
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n)
    return {"a_only_correct": b, "b_only_correct": c, "p_value": min(1.0, 2 * tail)}


def main():
    parser = argparse.ArgumentParser(description="Paired statistical comparison of experiment variants")
    parser.add_argument("--root", required=True)
    parser.add_argument("--variant-a", required=True)
    parser.add_argument("--variant-b", required=True)
    parser.add_argument(
        "--evaluation",
        choices=["fixed_0_5", "f1", "target_fpr", "target_fpr_confident"],
        default="target_fpr_confident",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--bootstrap-unit", choices=["site", "page"], default="site"
    )
    parser.add_argument("--cluster-column", default="codebase_id")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cluster_column = args.cluster_column if args.bootstrap_unit == "site" else None

    aligned = align(
        load_variant(Path(args.root), args.variant_a, args.evaluation),
        load_variant(Path(args.root), args.variant_b, args.evaluation),
        cluster_column=cluster_column,
    )
    rng = np.random.default_rng(args.seed)
    rows = []
    for metric, function in METRICS.items():
        observed = []
        draws = []
        for _seed, y, pa, pb, ta, tb, _clusters in aligned:
            observed.append(float(function(y, pb, tb) - function(y, pa, ta)))
        for _ in range(args.bootstrap_samples):
            seed_deltas = []
            for _seed, y, pa, pb, ta, tb, clusters in aligned:
                sample_frame = pd.DataFrame({"label": y})
                if cluster_column:
                    sample_frame[cluster_column] = clusters
                idx = stratified_bootstrap_indices(
                    sample_frame, rng, cluster_column=cluster_column
                )
                seed_deltas.append(float(function(y[idx], pb[idx], tb) - function(y[idx], pa[idx], ta)))
            draws.append(float(np.mean(seed_deltas)))
        rows.append({
            "metric": metric,
            "delta_b_minus_a": float(np.mean(observed)),
            "ci95_low": float(np.percentile(draws, 2.5)),
            "ci95_high": float(np.percentile(draws, 97.5)),
            "probability_b_better": float(np.mean(np.asarray(draws) > 0)),
        })
    mcnemar = [
        exact_mcnemar(y, pa, pb, ta, tb)
        for _seed, y, pa, pb, ta, tb, _clusters in aligned
    ]
    result = {
        "variant_a": args.variant_a,
        "variant_b": args.variant_b,
        "evaluation": args.evaluation,
        "common_seeds": [item[0] for item in aligned],
        "bootstrap_unit": args.bootstrap_unit,
        "cluster_column": cluster_column,
        "paired_bootstrap": rows,
        "mcnemar_by_seed": mcnemar,
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
