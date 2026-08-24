import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research import sha256_file, split_integrity, write_json


def duplicate_label_conflicts(frame):
    if "html_hash" not in frame:
        return 0
    counts = frame.groupby("html_hash")["label"].nunique()
    return int(counts.gt(1).sum())


def main():
    parser = argparse.ArgumentParser(description="Publication gate for dataset integrity")
    parser.add_argument("--scenario-dir", required=True)
    parser.add_argument("--min-test-positives", type=int, default=100)
    parser.add_argument("--min-test-positive-sites", type=int, default=0)
    parser.add_argument("--heldout-family", default="")
    parser.add_argument("--min-heldout-positives", type=int, default=0)
    parser.add_argument("--min-heldout-positive-sites", type=int, default=0)
    parser.add_argument("--min-heldout-positive-templates", type=int, default=0)
    parser.add_argument("--min-heldout-positive-brands", type=int, default=0)
    parser.add_argument("--require-template-isolation", action="store_true")
    parser.add_argument("--require-manual-llm-validation", action="store_true")
    parser.add_argument(
        "--allow-published-llm-labels",
        action="store_true",
        help="Treat published-study labels as acceptable provenance for LLM samples",
    )
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    root = Path(args.scenario_dir)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = {}
    paths = {}
    for name in ("train", "val", "test"):
        declared = manifest.get("files", {}).get(name, f"{name}.csv")
        paths[name] = root / declared
        frames[name] = pd.read_csv(paths[name], usecols=lambda column: column != "html")

    integrity = split_integrity(frames["train"], frames["val"], frames["test"])
    failures = []
    for pair, overlaps in integrity["overlap"].items():
        if overlaps.get("html_hash", 0):
            failures.append(f"{pair}: exact HTML overlap={overlaps['html_hash']}")
        if args.require_template_isolation and overlaps.get("template_id", 0):
            failures.append(f"{pair}: template overlap={overlaps['template_id']}")
    test_positives = int(pd.to_numeric(frames["test"]["label"]).eq(1).sum())
    if test_positives < args.min_test_positives:
        failures.append(
            f"test positives={test_positives}, required>={args.min_test_positives}"
        )
    heldout_positives = 0
    heldout_positive_sites = 0
    heldout_positive_templates = 0
    heldout_positive_brands = 0
    if args.heldout_family:
        if "generator_family" not in frames["test"]:
            failures.append("test is missing generator_family for held-out gate")
        else:
            family = frames["test"]["generator_family"].fillna("").astype(str).str.lower()
            heldout_mask = (
                family.eq(args.heldout_family.strip().lower()) &
                pd.to_numeric(frames["test"]["label"]).eq(1)
            )
            heldout_positives = int(heldout_mask.sum())
            if heldout_positives < args.min_heldout_positives:
                failures.append(
                    f"held-out {args.heldout_family} positives={heldout_positives}, "
                    f"required>={args.min_heldout_positives}"
                )
            heldout_positive = frames["test"].loc[heldout_mask]
            if "codebase_id" in heldout_positive:
                heldout_positive_sites = int(
                    heldout_positive["codebase_id"].fillna("").astype(str)
                    .replace("", pd.NA).nunique()
                )
            if "template_id" in heldout_positive:
                heldout_positive_templates = int(
                    heldout_positive["template_id"].fillna("").astype(str)
                    .replace("", pd.NA).nunique()
                )
            if "brand" in heldout_positive:
                brands = heldout_positive["brand"].fillna("").astype(str).str.strip().str.lower()
                heldout_positive_brands = int(
                    brands[~brands.isin({"", "unknown", "none", "nan"})].nunique()
                )
            for value, minimum, label in (
                (heldout_positive_sites, args.min_heldout_positive_sites, "sites"),
                (heldout_positive_templates, args.min_heldout_positive_templates, "templates"),
                (heldout_positive_brands, args.min_heldout_positive_brands, "brands"),
            ):
                if minimum and value < minimum:
                    failures.append(
                        f"held-out {args.heldout_family} positive {label}={value}, "
                        f"required>={minimum}"
                    )
    test_positive_sites = 0
    if "codebase_id" in frames["test"] and "site_label" in frames["test"]:
        site_labels = frames["test"].groupby("codebase_id")["site_label"]
        conflicts = int(site_labels.nunique(dropna=False).gt(1).sum())
        if conflicts:
            failures.append(f"test codebases with conflicting site_label={conflicts}")
        test_positive_sites = int(site_labels.first().astype(int).eq(1).sum())
    if args.min_test_positive_sites and test_positive_sites < args.min_test_positive_sites:
        failures.append(
            f"test positive sites={test_positive_sites}, required>={args.min_test_positive_sites}"
        )
    unvalidated_llm = {}
    if args.require_manual_llm_validation:
        for name, frame in frames.items():
            if "source" not in frame or "manual_validated" not in frame:
                continue
            llm_phishing = frame["source"].astype(str).str.lower().eq("llm") & pd.to_numeric(frame["label"]).eq(1)
            if "dataset_domain" in frame:
                llm_phishing |= frame["dataset_domain"].astype(str).str.lower().eq("llm_generated") & pd.to_numeric(frame["label"]).eq(1)
            manual = pd.to_numeric(frame["manual_validated"], errors="coerce").fillna(0).eq(1)
            acceptable = manual
            if args.allow_published_llm_labels and "published_label" in frame:
                acceptable |= pd.to_numeric(frame["published_label"], errors="coerce").fillna(0).eq(1)
            unvalidated_llm[name] = int((llm_phishing & ~acceptable).sum())
        if any(unvalidated_llm.values()):
            failures.append(f"LLM phishing without manual validation: {unvalidated_llm}")
    conflicts = {
        name: duplicate_label_conflicts(frame) for name, frame in frames.items()
    }
    if any(conflicts.values()):
        failures.append(f"conflicting labels by HTML hash: {conflicts}")

    report = {
        "passed": not failures,
        "scenario_dir": str(root.resolve()),
        "files": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
        "integrity": integrity,
        "duplicate_label_conflicts": conflicts,
        "publication_gate": {
            "min_test_positives": args.min_test_positives,
            "min_test_positive_sites": args.min_test_positive_sites,
            "test_positive_sites": test_positive_sites,
            "heldout_family": args.heldout_family,
            "min_heldout_positives": args.min_heldout_positives,
            "heldout_positives": heldout_positives,
            "min_heldout_positive_sites": args.min_heldout_positive_sites,
            "heldout_positive_sites": heldout_positive_sites,
            "min_heldout_positive_templates": args.min_heldout_positive_templates,
            "heldout_positive_templates": heldout_positive_templates,
            "min_heldout_positive_brands": args.min_heldout_positive_brands,
            "heldout_positive_brands": heldout_positive_brands,
            "require_template_isolation": args.require_template_isolation,
            "require_manual_llm_validation": args.require_manual_llm_validation,
            "allow_published_llm_labels": args.allow_published_llm_labels,
            "unvalidated_llm_phishing": unvalidated_llm,
            "failures": failures,
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        write_json(args.out, report)
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
