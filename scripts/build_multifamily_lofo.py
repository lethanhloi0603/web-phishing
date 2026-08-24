"""Build leakage-controlled LOFO scenarios for several LLM families."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research import sha256_file, source_snapshot


DEFAULT_FAMILIES = ("claude", "gemini", "gpt", "grok", "mistral", "qwen")


def parse_families(value):
    families = []
    for item in str(value).split(","):
        family = item.strip().lower()
        if family and family not in families:
            families.append(family)
    if not families:
        raise ValueError("At least one held-out family is required")
    return families


def family_counts(path):
    frame = pd.read_csv(path, usecols=["generator_family", "label"])
    frame["generator_family"] = (
        frame["generator_family"].fillna("").astype(str).str.strip().str.lower()
    )
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    return {
        str(family): {
            "negative": int(labels.eq(0).sum()),
            "positive": int(labels.eq(1).sum()),
            "total": int(len(labels)),
        }
        for family, labels in frame.groupby("generator_family")["label"]
    }


def heldout_counts(test_path, family):
    frame = pd.read_csv(test_path, usecols=lambda column: column != "html")
    selected = frame[
        frame["generator_family"].fillna("").astype(str).str.lower().eq(family)
    ].copy()
    labels = pd.to_numeric(selected["label"], errors="coerce")
    positive = selected[labels.eq(1)]
    return {
        "negative": int(labels.eq(0).sum()),
        "positive": int(labels.eq(1).sum()),
        "total": int(len(selected)),
        "positive_sites": int(
            positive.get("codebase_id", pd.Series(dtype=str))
            .fillna("").astype(str).replace("", pd.NA).nunique()
        ),
        "positive_templates": int(
            positive.get("template_id", pd.Series(dtype=str))
            .fillna("").astype(str).replace("", pd.NA).nunique()
        ),
    }


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(
        description="Build one strict LOFO scenario per eligible LLM family"
    )
    parser.add_argument("--phresh-train", required=True)
    parser.add_argument("--phresh-test", required=True)
    parser.add_argument("--llm", required=True)
    parser.add_argument("--llm-audit", default="")
    parser.add_argument(
        "--families", default=",".join(DEFAULT_FAMILIES),
        help="Comma-separated held-out families",
    )
    parser.add_argument(
        "--label-policy", choices=["manual", "published"], default="manual",
        help="Use manual labels for final evidence or source-published labels for pilots",
    )
    parser.add_argument("--min-positive-pages", type=int, default=20)
    parser.add_argument("--min-negative-pages", type=int, default=20)
    parser.add_argument("--validation-size", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-human-phishing-test",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--strict-template-isolation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--compression", choices=["none", "gzip"], default="gzip")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    families = parse_families(args.families)
    observed = family_counts(args.llm)
    problems = []
    for family in families:
        counts = observed.get(family, {"positive": 0, "negative": 0})
        if counts["positive"] < args.min_positive_pages:
            problems.append(
                f"{family}: positive={counts['positive']}, required>={args.min_positive_pages}"
            )
        if counts["negative"] < args.min_negative_pages:
            problems.append(
                f"{family}: negative={counts['negative']}, required>={args.min_negative_pages}"
            )
    if problems:
        raise ValueError("Ineligible LOFO families: " + "; ".join(problems))

    output_root = Path(args.out_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "multifamily_manifest.json"
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label_policy": args.label_policy,
        "families": families,
        "validation_size": args.validation_size,
        "seed": args.seed,
        "strict_template_isolation": args.strict_template_isolation,
        "include_human_phishing_test": args.include_human_phishing_test,
        "source_snapshot": source_snapshot(ROOT),
        "inputs": {
            "phresh_train": {
                "path": str(Path(args.phresh_train).resolve()),
                "sha256": sha256_file(args.phresh_train),
            },
            "phresh_test": {
                "path": str(Path(args.phresh_test).resolve()),
                "sha256": sha256_file(args.phresh_test),
            },
            "llm": {
                "path": str(Path(args.llm).resolve()),
                "sha256": sha256_file(args.llm),
            },
        },
        "source_family_counts": observed,
        "folds": {},
    }
    if args.llm_audit:
        manifest["inputs"]["llm_audit"] = {
            "path": str(Path(args.llm_audit).resolve()),
            "sha256": sha256_file(args.llm_audit),
        }
    atomic_json(manifest_path, manifest)

    for family in families:
        fold_dir = output_root / family
        if (
            fold_dir.exists()
            and any(fold_dir.iterdir())
            and not args.overwrite
        ):
            raise FileExistsError(
                f"{fold_dir} is not empty; pass --overwrite to rebuild it"
            )
        command = [
            sys.executable,
            str(ROOT / "scripts" / "build_scenarios.py"),
            "--phresh-train", args.phresh_train,
            "--phresh-test", args.phresh_test,
            "--llm", args.llm,
            "--require-validated",
            "--scenario", "lofo",
            "--holdout-family", family,
            "--validation-size", str(args.validation_size),
            "--seed", str(args.seed),
            "--compression", args.compression,
            "--out-dir", str(fold_dir),
        ]
        if args.llm_audit:
            command.extend(["--llm-audit", args.llm_audit])
        if args.label_policy == "published":
            command.append("--allow-published-labels")
        if args.include_human_phishing_test:
            command.append("--include-human-phishing-test")
        if args.strict_template_isolation:
            command.append("--strict-template-isolation")

        fold_record = {
            "state": "building",
            "scenario_dir": str(fold_dir),
            "command": [str(value) for value in command],
        }
        manifest["folds"][family] = fold_record
        atomic_json(manifest_path, manifest)
        try:
            subprocess.run(command, cwd=ROOT, check=True)
            scenario_manifest = json.loads(
                (fold_dir / "manifest.json").read_text(encoding="utf-8")
            )
            test_name = scenario_manifest.get("files", {}).get(
                "test", "test.csv.gz"
            )
            final_counts = heldout_counts(fold_dir / test_name, family)
            if final_counts["positive"] < args.min_positive_pages:
                raise ValueError(
                    f"{family}: only {final_counts['positive']} positives remain after isolation"
                )
            if final_counts["negative"] < args.min_negative_pages:
                raise ValueError(
                    f"{family}: only {final_counts['negative']} negatives remain after isolation"
                )
            fold_record.update({
                "state": "complete",
                "heldout_counts": final_counts,
                "scenario_manifest_sha256": sha256_file(fold_dir / "manifest.json"),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as exc:
            fold_record.update({
                "state": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
            atomic_json(manifest_path, manifest)
            raise
        atomic_json(manifest_path, manifest)

    print(f"Multi-family LOFO scenarios: {output_root}")


if __name__ == "__main__":
    main()
