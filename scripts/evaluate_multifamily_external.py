"""Evaluate every LOFO family's primary checkpoints on one external dataset."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def resolve_split(scenario_dir, name):
    manifest = json.loads(
        (scenario_dir / "manifest.json").read_text(encoding="utf-8")
    )
    declared = manifest.get("files", {}).get(name, f"{name}.csv.gz")
    path = scenario_dir / declared
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    value["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(
        description="Run provenance-disjoint external evaluation for all LOFO folds"
    )
    parser.add_argument("--scenario-root", required=True)
    parser.add_argument("--checkpoints-root", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--families", default="")
    parser.add_argument("--variant", default="robust_full")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--fpr-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--min-external-sites-per-class", type=int, default=100)
    parser.add_argument("--min-free-gb", type=float, default=5.5)
    parser.add_argument("--artifact-tests", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    scenario_root = Path(args.scenario_root).resolve()
    checkpoints_root = Path(args.checkpoints_root).resolve()
    output_root = Path(args.output_root).resolve()
    multifamily = json.loads(
        (scenario_root / "multifamily_manifest.json").read_text(encoding="utf-8")
    )
    families = [
        value.strip().lower() for value in args.families.split(",") if value.strip()
    ] or list(multifamily.get("families", []))
    if not families:
        raise ValueError("No LOFO families were declared")

    manifest_path = output_root / "multifamily_external_manifest.json"
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenario_root": str(scenario_root),
        "checkpoints_root": str(checkpoints_root),
        "external_test": str(Path(args.test).resolve()),
        "families": families,
        "variant": args.variant,
        "seeds": args.seeds,
        "folds": {},
    }
    atomic_json(manifest_path, manifest)

    for family in families:
        scenario_dir = scenario_root / family
        val = resolve_split(scenario_dir, "val")
        family_output = output_root / family
        command = [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_external_matrix.py"),
            "--checkpoints-root", str(checkpoints_root / family),
            "--variant", args.variant,
            "--seeds", args.seeds,
            "--val", str(val),
            "--test", args.test,
            "--output-dir", str(family_output),
            "--batch-size", str(args.batch_size),
            "--target-fpr", str(args.target_fpr),
            "--fpr-confidence", str(args.fpr_confidence),
            "--bootstrap-samples", str(args.bootstrap_samples),
            "--min-external-sites-per-class",
            str(args.min_external_sites_per_class),
            "--min-free-gb", str(args.min_free_gb),
        ]
        if args.artifact_tests:
            command.append("--artifact-tests")
        if args.overwrite:
            command.append("--overwrite")
        record = {
            "state": "running",
            "validation": str(val),
            "output_dir": str(family_output),
            "command": [str(value) for value in command],
        }
        manifest["folds"][family] = record
        atomic_json(manifest_path, manifest)
        try:
            subprocess.run(command, cwd=ROOT, check=True)
            record.update({
                "state": "complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as exc:
            record.update({
                "state": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
            atomic_json(manifest_path, manifest)
            raise
        atomic_json(manifest_path, manifest)

    print(f"Multi-family external evaluation: {output_root}")


if __name__ == "__main__":
    main()
