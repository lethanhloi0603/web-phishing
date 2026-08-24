"""Fail-closed evidence gate for the multi-family LOFO study design."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research import write_json


def add_issue(collection, code, message, observed=None, required=None):
    issue = {"code": code, "message": message}
    if observed is not None:
        issue["observed"] = observed
    if required is not None:
        issue["required"] = required
    collection.append(issue)


def parse_seeds(value):
    seeds = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be a non-empty unique comma-separated list")
    return seeds


def resolve_split(scenario_dir, name):
    manifest = json.loads(
        (scenario_dir / "manifest.json").read_text(encoding="utf-8")
    )
    path = scenario_dir / manifest.get("files", {}).get(name, f"{name}.csv.gz")
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def heldout_evidence(scenario_root, families):
    evidence = {}
    ownership = set()
    duplicates = set()
    for family in families:
        test_path = resolve_split(scenario_root / family, "test")
        frame = pd.read_csv(test_path, usecols=lambda column: column != "html")
        selected = frame[
            frame["generator_family"].fillna("").astype(str).str.lower().eq(family)
        ].copy()
        if "dataset_domain" in selected:
            selected = selected[
                selected["dataset_domain"].fillna("").astype(str).str.lower()
                .eq("llm_generated")
            ]
        sample_ids = set(selected["sample_id"].astype(str))
        duplicates.update(sample_ids & ownership)
        ownership.update(sample_ids)
        labels = pd.to_numeric(selected["label"], errors="coerce")
        positive = selected[labels.eq(1)]
        brands = positive.get("brand", pd.Series(dtype=str)).fillna("").astype(str)
        brands = brands.str.strip().str.lower()
        evidence[family] = {
            "rows": int(len(selected)),
            "positive": int(labels.eq(1).sum()),
            "negative": int(labels.eq(0).sum()),
            "positive_sites": int(
                positive.get("codebase_id", pd.Series(dtype=str))
                .fillna("").astype(str).replace("", pd.NA).nunique()
            ),
            "positive_templates": int(
                positive.get("template_id", pd.Series(dtype=str))
                .fillna("").astype(str).replace("", pd.NA).nunique()
            ),
            "positive_brands": int(
                brands[~brands.isin({"", "unknown", "none", "nan"})].nunique()
            ),
            "brands": sorted(
                brands[~brands.isin({"", "unknown", "none", "nan"})]
                .unique().tolist()
            ),
            "manual_positive": int(
                pd.to_numeric(
                    positive.get("manual_validated", pd.Series(0, index=positive.index)),
                    errors="coerce",
                ).fillna(0).eq(1).sum()
            ),
        }
    return evidence, sorted(duplicates)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-family LOFO publication evidence gate"
    )
    parser.add_argument("--scenario-root", required=True)
    parser.add_argument("--experiments-root", required=True)
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--agreement-json", required=True)
    parser.add_argument("--variant", default="robust_full")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--min-families", type=int, default=6)
    parser.add_argument("--min-total-positive", type=int, default=150)
    parser.add_argument("--min-total-negative", type=int, default=150)
    parser.add_argument("--min-family-positive", type=int, default=20)
    parser.add_argument("--min-family-negative", type=int, default=20)
    parser.add_argument("--min-family-positive-sites", type=int, default=15)
    parser.add_argument("--min-family-positive-templates", type=int, default=20)
    parser.add_argument("--min-macro-recall", type=float, default=0.5)
    parser.add_argument("--min-worst-family-recall", type=float, default=0.3)
    parser.add_argument("--out", default="outputs/audits/multifamily_q1_readiness.json")
    parser.add_argument("--advisory", action="store_true")
    args = parser.parse_args()

    scenario_root = Path(args.scenario_root).resolve()
    experiments_root = Path(args.experiments_root).resolve()
    external_root = Path(args.external_root).resolve()
    multifamily = json.loads(
        (scenario_root / "multifamily_manifest.json").read_text(encoding="utf-8")
    )
    families = list(multifamily.get("families", []))
    seeds = parse_seeds(args.seeds)
    blockers = []
    warnings = []
    if len(families) < args.min_families:
        add_issue(
            blockers, "family_coverage", "Too few primary LOFO families",
            len(families), args.min_families,
        )

    evidence, duplicate_samples = heldout_evidence(scenario_root, families)
    if duplicate_samples:
        add_issue(
            blockers, "oof_duplicate_samples",
            "Held-out samples appear in more than one family fold",
            duplicate_samples[:20], [],
        )
    total_positive = sum(item["positive"] for item in evidence.values())
    total_negative = sum(item["negative"] for item in evidence.values())
    if total_positive < args.min_total_positive:
        add_issue(
            blockers, "oof_positive_size", "Too few pooled OOF positives",
            total_positive, args.min_total_positive,
        )
    if total_negative < args.min_total_negative:
        add_issue(
            blockers, "oof_negative_size", "Too few pooled OOF negatives",
            total_negative, args.min_total_negative,
        )
    for family, item in evidence.items():
        for field, minimum in (
            ("positive", args.min_family_positive),
            ("negative", args.min_family_negative),
            ("positive_sites", args.min_family_positive_sites),
            ("positive_templates", args.min_family_positive_templates),
        ):
            if item[field] < minimum:
                add_issue(
                    blockers, f"family_{field}",
                    f"Held-out {family} has insufficient {field.replace('_', ' ')}",
                    item[field], minimum,
                )
        if item["manual_positive"] < item["positive"]:
            add_issue(
                blockers, "family_manual_validation",
                f"Held-out {family} positives are not all independently validated",
                item["manual_positive"], item["positive"],
            )
    known_brand_names = sorted({
        brand for item in evidence.values() for brand in item.get("brands", [])
    })
    known_brands = len(known_brand_names)
    if known_brands < 3:
        add_issue(
            warnings, "llm_brand_coverage",
            "The LLM-generated benchmark remains narrow in brand coverage",
            known_brands, "report as a limitation and use cross-brand external evidence",
        )

    fold_reports = {}
    for family in families:
        primary_metrics = [
            experiments_root / family / args.variant / f"seed_{seed}" / "metrics.json"
            for seed in seeds
        ]
        external_metrics = [
            external_root / family / args.variant / f"seed_{seed}" / "metrics.json"
            for seed in seeds
        ]
        fold_report_path = Path(args.out).resolve().parent / f"q1_{family}.json"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "q1_readiness.py"),
            "--scenario-dir", str(scenario_root / family),
            "--heldout-family", family,
            "--min-heldout-positives", str(args.min_family_positive),
            "--min-heldout-positive-sites", str(args.min_family_positive_sites),
            "--min-heldout-positive-templates", str(args.min_family_positive_templates),
            "--min-heldout-positive-brands", "0",
            "--agreement-json", args.agreement_json,
            "--min-seeds", str(len(seeds)),
            "--min-external-seeds", str(len(seeds)),
            "--metrics", *[str(path) for path in primary_metrics],
            "--external-metrics", *[str(path) for path in external_metrics],
            "--out", str(fold_report_path),
            "--advisory",
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        report = json.loads(fold_report_path.read_text(encoding="utf-8"))
        fold_reports[family] = {
            "path": str(fold_report_path),
            "policy_passed": bool(report.get("policy_passed")),
            "blockers": report.get("blockers", []),
        }
        if not report.get("policy_passed"):
            add_issue(
                blockers, "fold_evidence",
                f"LOFO evidence gate failed for {family}",
                [item.get("code") for item in report.get("blockers", [])], [],
            )

    summary_path = Path(args.summary_json)
    if not summary_path.exists():
        add_issue(
            blockers, "multifamily_summary_missing",
            "Multi-family summary JSON does not exist", str(summary_path), "existing file",
        )
        summary_report = {}
    else:
        summary_report = json.loads(summary_path.read_text(encoding="utf-8"))
    if set(summary_report.get("families", [])) != set(families):
        add_issue(
            blockers, "summary_family_identity",
            "Summary family set does not match the scenario manifest",
            summary_report.get("families", []), families,
        )
    if set(summary_report.get("seeds", [])) != set(seeds):
        add_issue(
            blockers, "summary_seed_identity",
            "Summary seeds do not match the declared primary seeds",
            summary_report.get("seeds", []), seeds,
        )
    summary_rows = pd.DataFrame(summary_report.get("summary", []))
    macro_recall = None
    if not summary_rows.empty:
        selected = summary_rows[
            summary_rows["variant"].eq(args.variant)
            & summary_rows["level"].eq("page")
            & summary_rows["scope"].eq("macro_family")
            & summary_rows["metric"].eq("recall")
        ]
        if len(selected) == 1:
            macro_recall = float(selected.iloc[0]["mean"])
    if macro_recall is None or macro_recall < args.min_macro_recall:
        add_issue(
            blockers, "macro_recall",
            "Macro-family page recall is missing or below policy",
            macro_recall, args.min_macro_recall,
        )
    aggregate_path = summary_path.parent / "aggregate_seed.csv"
    worst_recall = None
    if aggregate_path.exists():
        aggregate = pd.read_csv(aggregate_path)
        selected = aggregate[
            aggregate["variant"].eq(args.variant)
            & aggregate["level"].eq("page")
            & aggregate["scope"].eq("macro_family")
        ]
        if not selected.empty:
            worst_recall = float(pd.to_numeric(selected["worst_recall"]).min())
    if worst_recall is None or worst_recall < args.min_worst_family_recall:
        add_issue(
            blockers, "worst_family_recall",
            "Worst-family page recall is missing or below policy",
            worst_recall, args.min_worst_family_recall,
        )

    report = {
        "policy_passed": not blockers,
        "scope_note": "Project evidence policy; journal acceptance is never guaranteed.",
        "design": "multi-family leave-one-family-out",
        "families": families,
        "seeds": seeds,
        "heldout_evidence": evidence,
        "oof_total_positive": total_positive,
        "oof_total_negative": total_negative,
        "macro_recall": macro_recall,
        "worst_family_recall": worst_recall,
        "fold_reports": fold_reports,
        "blockers": blockers,
        "warnings": warnings,
    }
    write_json(args.out, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["policy_passed"] or args.advisory else 2)


if __name__ == "__main__":
    main()
