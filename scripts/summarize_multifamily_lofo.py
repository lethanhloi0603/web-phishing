"""Summarize out-of-family predictions across LOFO families and seeds."""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import binary_metrics, clopper_pearson_upper
from src.site_evaluation import aggregate_site_predictions


METRICS = (
    "precision", "recall", "f1", "auprc", "auroc", "fpr", "fnr",
    "mcc", "brier", "ece",
)


def parse_values(value):
    values = []
    for item in str(value).split(","):
        item = item.strip()
        if item and item not in values:
            values.append(item)
    return values


def threshold_from_report(report, evaluation, level="page"):
    if evaluation == "fixed_0_5":
        return 0.5
    section = report if level == "page" else report.get("site_level", {})
    try:
        return float(section["calibration"][evaluation]["threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Missing {level}-level calibration '{evaluation}'"
        ) from exc


def metrics_from_decisions(labels, probability, prediction):
    labels = np.asarray(labels, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = np.asarray(prediction, dtype=int)
    result = binary_metrics(labels, probability, threshold=0.5)
    tn, fp, fn, tp = confusion_matrix(
        labels, prediction, labels=[0, 1]
    ).ravel()
    result.update({
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if fp + tn else float("nan"),
        "fpr_upper_95": clopper_pearson_upper(fp, fp + tn, confidence=0.95),
        "fnr": float(fn / (fn + tp)) if fn + tp else float("nan"),
        "fnr_upper_95": clopper_pearson_upper(fn, fn + tp, confidence=0.95),
        "mcc": float(matthews_corrcoef(labels, prediction)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    })
    return result


def select_heldout(frame, family):
    if "generator_family" not in frame:
        raise ValueError("Predictions are missing generator_family")
    selected = frame[
        frame["generator_family"].fillna("").astype(str).str.strip().str.lower()
        .eq(family)
    ].copy()
    if "dataset_domain" in selected:
        selected = selected[
            selected["dataset_domain"].fillna("").astype(str).str.lower()
            .eq("llm_generated")
        ].copy()
    if selected.empty:
        raise ValueError(f"No out-of-family predictions found for '{family}'")
    labels = pd.to_numeric(selected["label"], errors="raise").astype(int)
    if set(labels.unique()) != {0, 1}:
        raise ValueError(f"Held-out family '{family}' must contain both labels")
    selected["label"] = labels
    return selected.reset_index(drop=True)


def evaluate_level(frame, threshold, level):
    evaluated = frame.copy()
    if level == "site":
        evaluated = aggregate_site_predictions(evaluated, aggregation="max")
    probability = evaluated["probability"].astype(float).to_numpy()
    labels = evaluated["label"].astype(int).to_numpy()
    prediction = (probability >= threshold).astype(int)
    return evaluated, metrics_from_decisions(labels, probability, prediction)


def macro_bootstrap(frame, metric, samples, rng):
    families = sorted(frame["family"].unique())
    seeds = sorted(frame["seed"].unique())
    matrix = frame.pivot(index="family", columns="seed", values=metric)
    matrix = matrix.loc[families, seeds].to_numpy(dtype=float)
    draws = []
    for _ in range(samples):
        family_index = rng.integers(0, len(families), size=len(families))
        seed_index = rng.integers(0, len(seeds), size=len(seeds))
        draws.append(float(np.nanmean(matrix[np.ix_(family_index, seed_index)])))
    return {
        "low": float(np.nanpercentile(draws, 2.5)),
        "high": float(np.nanpercentile(draws, 97.5)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create macro/micro/worst-family LOFO result tables"
    )
    parser.add_argument("--scenario-root", required=True)
    parser.add_argument("--experiments-root", required=True)
    parser.add_argument(
        "--families", default="",
        help="Optional subset for pilot summaries; final evidence uses all declared families",
    )
    parser.add_argument("--variants", default="bce,robust_full")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument(
        "--evaluation",
        choices=["fixed_0_5", "f1", "target_fpr", "target_fpr_confident"],
        default="target_fpr_confident",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    scenario_root = Path(args.scenario_root).resolve()
    experiments_root = Path(args.experiments_root).resolve()
    scenario_manifest = json.loads(
        (scenario_root / "multifamily_manifest.json").read_text(encoding="utf-8")
    )
    declared_families = list(scenario_manifest.get("families", []))
    families = parse_values(args.families) if args.families else declared_families
    unknown_families = sorted(set(families) - set(declared_families))
    if unknown_families:
        raise ValueError(f"Families are not declared in the scenario: {unknown_families}")
    variants = parse_values(args.variants)
    seeds = [int(value) for value in parse_values(args.seeds)]
    if len(families) < 2:
        raise ValueError("Multi-family summary requires at least two LOFO families")
    if not variants or not seeds:
        raise ValueError("At least one variant and seed are required")

    per_family_rows = []
    evaluated_frames = {}
    sample_ownership = {}
    for variant in variants:
        for seed in seeds:
            for family in families:
                run_dir = experiments_root / family / variant / f"seed_{seed}"
                metrics_path = run_dir / "metrics.json"
                predictions_path = run_dir / "predictions.csv"
                if not metrics_path.exists() or not predictions_path.exists():
                    raise FileNotFoundError(
                        f"Missing completed run for {family}/{variant}/seed_{seed}"
                    )
                report = json.loads(metrics_path.read_text(encoding="utf-8"))
                predictions = select_heldout(pd.read_csv(predictions_path), family)
                ownership_key = (variant, seed)
                sample_ids = set(predictions["sample_id"].astype(str))
                duplicate = sample_ids & sample_ownership.setdefault(ownership_key, set())
                if duplicate:
                    raise ValueError(
                        "Out-of-family samples appear in multiple folds: "
                        + ", ".join(sorted(duplicate)[:5])
                    )
                sample_ownership[ownership_key].update(sample_ids)

                for level in ("page", "site"):
                    threshold = threshold_from_report(report, args.evaluation, level=level)
                    evaluated, result = evaluate_level(predictions, threshold, level)
                    evaluated = evaluated.copy()
                    evaluated["_decision"] = (
                        evaluated["probability"].astype(float) >= threshold
                    ).astype(int)
                    evaluated["_family"] = family
                    evaluated_frames[(variant, seed, family, level)] = evaluated
                    row = {
                        "variant": variant,
                        "seed": seed,
                        "family": family,
                        "level": level,
                        "evaluation": args.evaluation,
                        "threshold": threshold,
                        "n": int(len(evaluated)),
                    }
                    row.update({metric: result[metric] for metric in METRICS})
                    per_family_rows.append(row)

    per_family = pd.DataFrame(per_family_rows)
    aggregate_rows = []
    for (variant, seed, level), group in per_family.groupby(
        ["variant", "seed", "level"], sort=True
    ):
        macro = {
            metric: float(pd.to_numeric(group[metric], errors="coerce").mean())
            for metric in METRICS
        }
        worst_recall_index = pd.to_numeric(group["recall"]).idxmin()
        valid_auprc = pd.to_numeric(group["auprc"], errors="coerce").dropna()
        worst_auprc_index = valid_auprc.idxmin() if not valid_auprc.empty else None
        aggregate_rows.append({
            "variant": variant,
            "seed": int(seed),
            "level": level,
            "scope": "macro_family",
            "n": int(group["n"].sum()),
            "worst_recall_family": str(group.loc[worst_recall_index, "family"]),
            "worst_recall": float(group.loc[worst_recall_index, "recall"]),
            "worst_auprc_family": (
                str(group.loc[worst_auprc_index, "family"])
                if worst_auprc_index is not None else ""
            ),
            "worst_auprc": (
                float(group.loc[worst_auprc_index, "auprc"])
                if worst_auprc_index is not None else float("nan")
            ),
            **macro,
        })

        combined = pd.concat([
            evaluated_frames[(variant, seed, family, level)]
            for family in families
        ], ignore_index=True)
        micro = metrics_from_decisions(
            combined["label"].astype(int).to_numpy(),
            combined["probability"].astype(float).to_numpy(),
            combined["_decision"].astype(int).to_numpy(),
        )
        aggregate_rows.append({
            "variant": variant,
            "seed": int(seed),
            "level": level,
            "scope": "micro_oof",
            "n": int(len(combined)),
            "worst_recall_family": "",
            "worst_recall": float("nan"),
            "worst_auprc_family": "",
            "worst_auprc": float("nan"),
            **{metric: micro[metric] for metric in METRICS},
        })

    aggregate = pd.DataFrame(aggregate_rows)
    rng = np.random.default_rng(args.bootstrap_seed)
    summary_rows = []
    for (variant, level), group in per_family.groupby(["variant", "level"]):
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            interval = macro_bootstrap(
                group, metric, args.bootstrap_samples, rng
            )
            summary_rows.append({
                "variant": variant,
                "level": level,
                "scope": "macro_family",
                "metric": metric,
                "families": len(families),
                "seeds": len(seeds),
                "mean": float(values.mean()),
                "std_across_family_seed": float(values.std(ddof=1)),
                "hierarchical_ci95_low": interval["low"],
                "hierarchical_ci95_high": interval["high"],
            })
    for (variant, level), group in aggregate[
        aggregate["scope"].eq("micro_oof")
    ].groupby(["variant", "level"]):
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            summary_rows.append({
                "variant": variant,
                "level": level,
                "scope": "micro_oof",
                "metric": metric,
                "families": len(families),
                "seeds": len(seeds),
                "mean": float(values.mean()),
                "std_across_family_seed": float(values.std(ddof=1)),
                "hierarchical_ci95_low": float("nan"),
                "hierarchical_ci95_high": float("nan"),
            })
    summary = pd.DataFrame(summary_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_family.to_csv(out_dir / "per_family_seed.csv", index=False)
    aggregate.to_csv(out_dir / "aggregate_seed.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    evidence = {
        "schema_version": 1,
        "scenario_root": str(scenario_root),
        "experiments_root": str(experiments_root),
        "families": families,
        "variants": variants,
        "seeds": seeds,
        "evaluation": args.evaluation,
        "bootstrap": {
            "method": "hierarchical family-and-seed bootstrap",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "oof_counts": {
            variant: {
                str(seed): int(len(sample_ownership[(variant, seed)]))
                for seed in seeds
            }
            for variant in variants
        },
        "files": {
            "per_family_seed": "per_family_seed.csv",
            "aggregate_seed": "aggregate_seed.csv",
            "summary": "summary.csv",
        },
        "summary": json.loads(summary.to_json(orient="records")),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"Multi-family summary: {out_dir}")


if __name__ == "__main__":
    main()
