"""Evidence gate for this project's Q1-oriented primary experiment.

This is a project policy, not a promise of journal acceptance. It deliberately
fails closed when required evidence is missing.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import clopper_pearson_upper
from src.research import sha256_file, split_integrity, write_json


def resolve_split(root, name):
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        declared = manifest.get("files", {}).get(name)
        if declared and (root / declared).exists():
            return root / declared
    for filename in (f"{name}.csv.gz", f"{name}.csv"):
        if (root / filename).exists():
            return root / filename
    raise FileNotFoundError(f"Missing {name} split under {root}")


def add_issue(collection, code, message, observed=None, required=None):
    issue = {"code": code, "message": message}
    if observed is not None:
        issue["observed"] = observed
    if required is not None:
        issue["required"] = required
    collection.append(issue)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def infer_seed(path, report):
    training_args = report.get("training_args", {})
    if "seed" in training_args:
        return str(training_args["seed"])
    match = re.search(r"seed[_-]?(\d+)", str(path), flags=re.IGNORECASE)
    return match.group(1) if match else str(Path(path).resolve())


def group_recall(report, family, section="groups_target_fpr_confident"):
    """Return recall for one generator family from a grouped metrics section."""
    family_groups = report.get(section, {}).get("generator_family", {})
    group = next(
        (
            values for name, values in family_groups.items()
            if str(name).strip().lower() == str(family).strip().lower()
        ),
        {},
    )
    value = group.get("recall", group.get("class_recall"))
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def primary_training_identity_issues(report, expected_hashes):
    """Validate that a result's checkpoint is bound to the declared train/val files."""
    issues = []
    if int(report.get("checkpoint_schema_version", 0) or 0) < 3:
        issues.append("checkpoint schema does not bind training inputs")
    observed = report.get("training_data_sha256")
    if not isinstance(observed, dict):
        observed = {}
    for split in ("train", "val"):
        if observed.get(split) != expected_hashes[split]:
            issues.append(
                f"checkpoint {split} SHA-256 does not match the declared scenario"
            )
    if not str(report.get("source_snapshot_sha256", "")).strip():
        issues.append("checkpoint source snapshot SHA-256 is missing")
    return issues


def training_configuration_signature(report):
    """Hash-equivalent JSON for hyperparameters that must match across seeds."""
    ignored = {
        "seed", "output", "overwrite", "train", "val", "min_free_gb",
        "log_every",
    }
    training_args = report.get("training_args", {})
    if not isinstance(training_args, dict):
        return ""
    comparable = {
        key: value for key, value in training_args.items() if key not in ignored
    }
    return json.dumps(comparable, sort_keys=True, separators=(",", ":"))


def validate_external_report(
    path,
    report,
    primary_provenance,
    primary_test_sha256,
    min_positives=100,
    min_negatives=100,
    primary_checkpoint_hashes=None,
    min_positive_sites=100,
    min_negative_sites=100,
):
    """Return fail-closed reasons why a metrics file is not independent evidence."""
    issues = []
    if report.get("evaluation_role") != "external":
        issues.append("evaluation_role must be explicitly set to 'external'")
    evidence = report.get("evaluation_data")
    if not isinstance(evidence, dict):
        return issues + ["evaluation_data metadata is missing"]
    if int(evidence.get("phishing", 0)) < min_positives:
        issues.append(
            f"external phishing rows={evidence.get('phishing', 0)}, required>={min_positives}"
        )
    if int(evidence.get("benign", 0)) < min_negatives:
        issues.append(
            f"external benign rows={evidence.get('benign', 0)}, required>={min_negatives}"
        )
    if int(evidence.get("phishing_sites", 0)) < min_positive_sites:
        issues.append(
            f"external phishing sites={evidence.get('phishing_sites', 0)}, "
            f"required>={min_positive_sites}"
        )
    if int(evidence.get("benign_sites", 0)) < min_negative_sites:
        issues.append(
            f"external benign sites={evidence.get('benign_sites', 0)}, "
            f"required>={min_negative_sites}"
        )
    if int(evidence.get("site_label_conflicts", 0)) > 0:
        issues.append(
            f"external data has {evidence.get('site_label_conflicts')} conflicting site labels"
        )
    provenance = {
        str(value).strip().lower()
        for value in evidence.get("dataset_provenance", [])
        if str(value).strip().lower() not in {"", "unknown", "none", "nan"}
    }
    if not provenance:
        issues.append("external dataset_provenance is missing")
    elif provenance & primary_provenance:
        issues.append(
            "external provenance overlaps the primary train/validation data: "
            + ", ".join(sorted(provenance & primary_provenance))
        )
    licenses = {
        str(value).strip().lower()
        for value in evidence.get("dataset_license", [])
        if str(value).strip().lower() not in {"", "unknown", "none", "nan"}
    }
    if not licenses or licenses <= {"noassertion", "unlicensed"}:
        issues.append(
            "external dataset has no declared redistribution/research license"
        )
    citations = {
        str(value).strip()
        for value in evidence.get("dataset_citation", [])
        if str(value).strip().lower() not in {"", "unknown", "none", "nan"}
    }
    if not citations:
        issues.append("external dataset citation/DOI is missing")
    if evidence.get("sha256") == primary_test_sha256:
        issues.append("external test split is identical to the primary test split")
    if primary_checkpoint_hashes is not None:
        checkpoint_hash = str(report.get("checkpoint_sha256", "")).strip()
        if not checkpoint_hash:
            issues.append("external result lacks checkpoint SHA-256")
        elif checkpoint_hash not in primary_checkpoint_hashes:
            issues.append("external result does not use a declared primary checkpoint")
    overall = report.get("overall", {})
    try:
        auprc = float(overall.get("auprc"))
    except (TypeError, ValueError):
        auprc = float("nan")
    if not math.isfinite(auprc):
        issues.append("external overall AUPRC is missing or invalid")
    return issues


def main():
    parser = argparse.ArgumentParser(description="Fail-closed Q1 evidence readiness report")
    parser.add_argument("--scenario-dir", required=True)
    parser.add_argument("--heldout-family", default="qwen")
    parser.add_argument("--min-heldout-positives", type=int, default=100)
    parser.add_argument("--min-heldout-positive-sites", type=int, default=50)
    parser.add_argument("--min-heldout-positive-templates", type=int, default=100)
    parser.add_argument("--min-heldout-positive-brands", type=int, default=3)
    parser.add_argument("--min-validation-negatives", type=int, default=299)
    parser.add_argument("--preferred-validation-negatives", type=int, default=1000)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--fpr-confidence", type=float, default=0.95)
    parser.add_argument("--min-target-recall", type=float, default=0.5)
    parser.add_argument("--min-heldout-recall", type=float, default=0.5)
    parser.add_argument("--min-heldout-site-recall", type=float, default=0.5)
    parser.add_argument("--artifact-fpr-limit", type=float, default=0.02)
    parser.add_argument("--min-artifact-recall", type=float, default=0.3)
    parser.add_argument("--min-artifact-heldout-recall", type=float, default=0.3)
    parser.add_argument("--agreement-json", default="")
    parser.add_argument("--min-double-annotated", type=int, default=100)
    parser.add_argument("--min-kappa", type=float, default=0.8)
    parser.add_argument("--metrics", nargs="*", default=[])
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--external-metrics", nargs="*", default=[])
    parser.add_argument("--min-external-positives", type=int, default=100)
    parser.add_argument("--min-external-negatives", type=int, default=100)
    parser.add_argument("--min-external-positive-sites", type=int, default=100)
    parser.add_argument("--min-external-negative-sites", type=int, default=100)
    parser.add_argument("--min-external-seeds", type=int, default=5)
    parser.add_argument("--out", default="outputs/audits/q1_readiness.json")
    parser.add_argument("--advisory", action="store_true", help="Write blockers but exit 0")
    args = parser.parse_args()

    root = Path(args.scenario_dir)
    frames = {
        name: pd.read_csv(resolve_split(root, name), usecols=lambda column: column != "html")
        for name in ("train", "val", "test")
    }
    blockers, warnings = [], []
    integrity = split_integrity(frames["train"], frames["val"], frames["test"])
    for pair, overlap in integrity["overlap"].items():
        if overlap.get("html_hash", 0):
            add_issue(blockers, "exact_leakage", f"{pair} has exact HTML overlap", overlap["html_hash"], 0)
        if overlap.get("template_id", 0):
            add_issue(blockers, "template_leakage", f"{pair} has template overlap", overlap["template_id"], 0)

    test = frames["test"]
    family = test.get("generator_family", pd.Series("", index=test.index)).fillna("").astype(str).str.lower()
    positive = pd.to_numeric(test["label"], errors="coerce").eq(1)
    heldout = test[family.eq(args.heldout_family.lower()) & positive].copy()
    counts = {
        "positive_pages": int(len(heldout)),
        "positive_sites": int(heldout.get("codebase_id", pd.Series(dtype=str)).fillna("").replace("", pd.NA).nunique()),
        "positive_templates": int(heldout.get("template_id", pd.Series(dtype=str)).fillna("").replace("", pd.NA).nunique()),
    }
    heldout_brands = heldout.get("brand", pd.Series(dtype=str)).fillna("").astype(str).str.strip().str.lower()
    counts["positive_brands"] = int(
        heldout_brands[~heldout_brands.isin({"", "unknown", "none", "nan"})].nunique()
    )
    for key, minimum in (
        ("positive_pages", args.min_heldout_positives),
        ("positive_sites", args.min_heldout_positive_sites),
        ("positive_templates", args.min_heldout_positive_templates),
        ("positive_brands", args.min_heldout_positive_brands),
    ):
        if counts[key] < minimum:
            add_issue(
                blockers, f"heldout_{key}",
                f"Held-out {args.heldout_family} has too few independent {key.replace('_', ' ')}",
                counts[key], minimum,
            )

    manual = pd.to_numeric(
        heldout.get("manual_validated", pd.Series(0, index=heldout.index)), errors="coerce"
    ).fillna(0).eq(1)
    manual_count = int(manual.sum())
    if manual_count < len(heldout):
        add_issue(
            blockers, "independent_manual_validation",
            "Published/source labels do not replace independent phishing-page validation",
            manual_count, int(len(heldout)),
        )

    val_negatives = int(pd.to_numeric(frames["val"]["label"], errors="coerce").eq(0).sum())
    zero_fp_upper = clopper_pearson_upper(0, val_negatives, args.fpr_confidence)
    if val_negatives < args.min_validation_negatives or zero_fp_upper > args.target_fpr:
        add_issue(
            blockers, "fpr_calibration_size",
            "Validation negatives cannot certify the target FPR even with zero false positives",
            {"negatives": val_negatives, "zero_fp_upper": zero_fp_upper},
            {"min_negatives": args.min_validation_negatives, "target_fpr": args.target_fpr},
        )
    elif val_negatives < args.preferred_validation_negatives:
        add_issue(
            warnings, "fpr_calibration_precision",
            "Validation can barely certify the target; more benign sites would narrow uncertainty",
            val_negatives, args.preferred_validation_negatives,
        )

    for split_name, frame in frames.items():
        if "source" not in frame:
            add_issue(blockers, "source_metadata", f"{split_name} lacks source metadata")
            continue
        source = frame["source"].fillna("").astype(str).str.lower().map(
            lambda value: "human" if value in {"human", "conventional", "real_world"}
            else ("llm" if value in {"llm", "llm_generated"} else value)
        )
        labels = pd.to_numeric(frame["label"], errors="coerce")
        for source_name in ("human", "llm"):
            observed = set(labels[source.eq(source_name)].dropna().astype(int).tolist())
            if observed != {0, 1}:
                add_issue(
                    blockers, "source_label_shortcut",
                    f"{split_name}/{source_name} does not contain both labels",
                    sorted(observed), [0, 1],
                )

    agreement = None
    if args.agreement_json:
        if not Path(args.agreement_json).exists():
            add_issue(
                blockers, "annotation_agreement_missing",
                f"Annotation agreement file does not exist: {args.agreement_json}",
            )
        else:
            try:
                agreement = load_json(args.agreement_json)
            except (OSError, json.JSONDecodeError) as exc:
                add_issue(
                    blockers, "annotation_agreement_invalid",
                    f"Cannot read annotation agreement evidence: {exc}",
                )
    if agreement is None:
        add_issue(blockers, "double_annotation", "No inter-rater agreement evidence was supplied")
    else:
        required_double = max(args.min_double_annotated, len(heldout))
        if int(agreement.get("n_double_annotated", 0)) < required_double:
            add_issue(blockers, "double_annotation_count", "Too few independently double-annotated samples", agreement.get("n_double_annotated", 0), required_double)
        if float(agreement.get("cohen_kappa", -1)) < args.min_kappa:
            add_issue(blockers, "annotation_agreement", "Cohen's kappa is below the project policy", agreement.get("cohen_kappa"), args.min_kappa)

    primary_split_sha256 = {
        split: sha256_file(resolve_split(root, split))
        for split in ("train", "val", "test")
    }
    primary_test_sha256 = primary_split_sha256["test"]
    metric_reports = []
    for path in args.metrics:
        if not Path(path).exists():
            add_issue(
                blockers, "primary_metrics_missing",
                f"Primary metrics file does not exist: {path}",
            )
            continue
        try:
            metric_reports.append((path, load_json(path)))
        except (OSError, json.JSONDecodeError) as exc:
            add_issue(
                blockers, "primary_metrics_invalid",
                f"Cannot read primary metrics {path}: {exc}",
            )
    seeds = {infer_seed(path, report) for path, report in metric_reports}
    if len(seeds) < args.min_seeds:
        add_issue(blockers, "multi_seed", "Primary model lacks enough independent seeds", len(seeds), args.min_seeds)
    source_snapshots = {
        str(report.get("source_snapshot_sha256", "")).strip()
        for _, report in metric_reports
        if str(report.get("source_snapshot_sha256", "")).strip()
    }
    if metric_reports and len(source_snapshots) != 1:
        add_issue(
            blockers, "source_snapshot_consistency",
            "Primary seeds were not produced from one recorded source snapshot",
            sorted(source_snapshots), "one shared snapshot",
        )
    training_signatures = {
        training_configuration_signature(report)
        for _, report in metric_reports
        if training_configuration_signature(report)
    }
    if metric_reports and len(training_signatures) != 1:
        add_issue(
            blockers, "training_configuration_consistency",
            "Primary seeds do not share one training configuration",
            len(training_signatures), 1,
        )
    for path, report in metric_reports:
        if report.get("evaluation_role") != "primary":
            add_issue(
                blockers, "primary_metrics_role",
                f"Primary result lacks evaluation_role=primary in {path}",
                report.get("evaluation_role"), "primary",
            )
        evaluation_data = report.get("evaluation_data", {})
        if evaluation_data.get("sha256") != primary_test_sha256:
            add_issue(
                blockers, "primary_test_identity",
                f"Primary result was not evaluated on the declared scenario test split: {path}",
                evaluation_data.get("sha256"), primary_test_sha256,
            )
        if not report.get("checkpoint_sha256"):
            add_issue(
                blockers, "checkpoint_identity",
                f"Primary result lacks checkpoint SHA-256: {path}",
            )
        identity_issues = primary_training_identity_issues(
            report, primary_split_sha256
        )
        if identity_issues:
            add_issue(
                blockers, "training_identity",
                f"Primary checkpoint is not reproducibly bound to the scenario in {path}",
                identity_issues, [],
            )
        confident = report.get("calibration", {}).get("target_fpr_confident")
        if not confident or not confident.get("validation", {}).get("target_fpr_certified", False):
            add_issue(blockers, "fpr_confidence_evidence", f"Missing certified target-FPR calibration in {path}")
        else:
            test_metrics = confident.get("test", {})
            if float(test_metrics.get("fpr_upper_95", 1.0)) > args.target_fpr:
                add_issue(
                    blockers, "test_fpr_confidence",
                    f"Test FPR upper bound exceeds the target in {path}",
                    test_metrics.get("fpr_upper_95"), args.target_fpr,
                )
            if float(test_metrics.get("recall", 0.0)) < args.min_target_recall:
                add_issue(
                    blockers, "target_fpr_recall",
                    f"Recall at the confidence-certified FPR is too low in {path}",
                    test_metrics.get("recall"), args.min_target_recall,
                )
        heldout_recall = group_recall(report, args.heldout_family)
        if heldout_recall is None:
            add_issue(
                blockers, "heldout_metrics_missing",
                f"Missing target-FPR metrics for held-out {args.heldout_family} in {path}",
            )
        elif heldout_recall < args.min_heldout_recall:
            add_issue(
                blockers, "heldout_recall",
                f"Held-out {args.heldout_family} page recall is too low in {path}",
                heldout_recall, args.min_heldout_recall,
            )
        heldout_site_recall = group_recall(
            report.get("site_level", {}), args.heldout_family
        )
        if heldout_site_recall is None:
            add_issue(
                blockers, "heldout_site_metrics_missing",
                f"Missing site-level target-FPR metrics for held-out {args.heldout_family} in {path}",
            )
        elif heldout_site_recall < args.min_heldout_site_recall:
            add_issue(
                blockers, "heldout_site_recall",
                f"Held-out {args.heldout_family} site recall is too low in {path}",
                heldout_site_recall, args.min_heldout_site_recall,
            )
        artifacts = report.get("artifact_tests", {})
        missing_artifacts = sorted(
            {"comments_removed", "identifiers_normalized", "visible_text_removed"} - set(artifacts)
        )
        if missing_artifacts:
            add_issue(blockers, "artifact_robustness", f"Missing artifact tests in {path}", missing_artifacts)
        for artifact_name, artifact_report in artifacts.items():
            artifact_metrics = artifact_report.get("target_fpr_confident", {})
            if float(artifact_metrics.get("fpr_upper_95", 1.0)) > args.artifact_fpr_limit:
                add_issue(
                    blockers, "artifact_fpr",
                    f"{artifact_name} FPR upper bound exceeds the robustness limit in {path}",
                    artifact_metrics.get("fpr_upper_95"), args.artifact_fpr_limit,
                )
            if float(artifact_metrics.get("recall", 0.0)) < args.min_artifact_recall:
                add_issue(
                    blockers, "artifact_recall",
                    f"{artifact_name} recall is below the robustness limit in {path}",
                    artifact_metrics.get("recall"), args.min_artifact_recall,
                )
            artifact_heldout_recall = group_recall(
                artifact_report, args.heldout_family
            )
            if artifact_heldout_recall is None:
                add_issue(
                    blockers, "artifact_heldout_metrics_missing",
                    f"{artifact_name} lacks held-out {args.heldout_family} metrics in {path}",
                )
            elif artifact_heldout_recall < args.min_artifact_heldout_recall:
                add_issue(
                    blockers, "artifact_heldout_recall",
                    f"{artifact_name} held-out {args.heldout_family} recall is too low in {path}",
                    artifact_heldout_recall, args.min_artifact_heldout_recall,
                )
    if not metric_reports:
        add_issue(blockers, "primary_metrics", "No primary-model metrics were supplied")

    primary_checkpoint_hashes = {
        str(report.get("checkpoint_sha256", "")).strip()
        for _, report in metric_reports
        if str(report.get("checkpoint_sha256", "")).strip()
    }
    if len(primary_checkpoint_hashes) < min(args.min_seeds, len(metric_reports)):
        add_issue(
            blockers, "checkpoint_independence",
            "Primary seed reports do not reference distinct checkpoints",
            len(primary_checkpoint_hashes), min(args.min_seeds, len(metric_reports)),
        )

    primary_provenance = set()
    for frame in (frames["train"], frames["val"]):
        if "dataset_provenance" in frame:
            primary_provenance.update(
                frame["dataset_provenance"].fillna("").astype(str).str.strip().str.lower()
            )
    primary_provenance -= {"", "unknown", "none", "nan"}
    valid_external, external_evidence, external_seeds = [], [], set()
    for path in args.external_metrics:
        if not Path(path).exists():
            add_issue(
                blockers, "external_metrics_missing",
                f"External metrics file does not exist: {path}",
            )
            continue
        try:
            external_report = load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            add_issue(
                blockers, "external_metrics_invalid",
                f"Cannot read external metrics {path}: {exc}",
            )
            continue
        issues = validate_external_report(
            path,
            external_report,
            primary_provenance,
            primary_test_sha256,
            args.min_external_positives,
            args.min_external_negatives,
            primary_checkpoint_hashes,
            min_positive_sites=args.min_external_positive_sites,
            min_negative_sites=args.min_external_negative_sites,
        )
        external_evidence.append({"path": path, "issues": issues})
        if issues:
            add_issue(
                blockers, "external_validation_invalid",
                f"External result is not publication-grade evidence: {path}",
                issues,
            )
        else:
            valid_external.append(path)
            external_seeds.add(infer_seed(path, external_report))
    if not valid_external:
        add_issue(blockers, "external_validation", "No independent external-dataset result was supplied")
    elif len(external_seeds) < args.min_external_seeds:
        add_issue(
            blockers, "external_multi_seed",
            "External validation lacks enough primary-model seeds",
            len(external_seeds), args.min_external_seeds,
        )

    report = {
        "policy_passed": not blockers,
        "scope_note": "Project evidence gate only; it cannot guarantee Q1 acceptance.",
        "scenario_dir": str(root.resolve()),
        "heldout_family": args.heldout_family,
        "heldout_counts": counts,
        "heldout_manually_validated": manual_count,
        "validation_negatives": val_negatives,
        "zero_fp_fpr_upper_bound": zero_fp_upper,
        "fpr_confidence": args.fpr_confidence,
        "metric_seeds": sorted(seeds),
        "external_metrics": valid_external,
        "external_evidence": external_evidence,
        "external_metric_seeds": sorted(external_seeds),
        "integrity": integrity,
        "blockers": blockers,
        "warnings": warnings,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    write_json(args.out, report)
    raise SystemExit(0 if report["policy_passed"] or args.advisory else 2)


if __name__ == "__main__":
    main()
