"""Run identical experiment matrices across multi-family LOFO folds."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    value["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def parse_values(value):
    values = []
    for item in str(value).split(","):
        item = item.strip()
        if item and item not in values:
            values.append(item)
    return values


def main():
    parser = argparse.ArgumentParser(
        description="Run the same variants and seeds for every LOFO family"
    )
    parser.add_argument("--scenario-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--families", default="")
    parser.add_argument("--variants", default="bce,robust_full")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-html-length", type=int, default=256)
    parser.add_argument("--max-dom-length", type=int, default=128)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--fpr-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--aux-warmup-epochs", type=float, default=1.0)
    parser.add_argument("--min-free-gb", type=float, default=5.5)
    parser.add_argument("--with-html", action="store_true")
    parser.add_argument("--freeze-html", action="store_true")
    parser.add_argument("--fusion-type", choices=["concat", "gated"], default="gated")
    parser.add_argument("--modality-dropout", type=float, default=0.1)
    parser.add_argument("--artifact-augmentation-probability", type=float, default=0.3)
    parser.add_argument(
        "--artifact-augmentation-modes",
        default="comments_removed,identifiers_normalized,visible_text_removed",
    )
    parser.add_argument(
        "--artifact-tests", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--artifact-test-variants", default="robust_full")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    scenario_root = Path(args.scenario_root).resolve()
    scenario_manifest_path = scenario_root / "multifamily_manifest.json"
    scenario_manifest = json.loads(
        scenario_manifest_path.read_text(encoding="utf-8")
    )
    declared_families = scenario_manifest.get("families", [])
    families = parse_values(args.families) if args.families else declared_families
    if not families:
        raise ValueError("No LOFO families were declared")
    missing = [
        family for family in families
        if scenario_manifest.get("folds", {}).get(family, {}).get("state") != "complete"
    ]
    if missing:
        raise ValueError(f"Scenario folds are not complete: {missing}")

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "multifamily_run_manifest.json"
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenario_root": str(scenario_root),
        "scenario_manifest": str(scenario_manifest_path),
        "families": families,
        "variants": parse_values(args.variants),
        "seeds": [int(value) for value in parse_values(args.seeds)],
        "settings": {
            key: value for key, value in vars(args).items()
            if key not in {"scenario_root", "output_root", "families"}
        },
        "folds": {},
    }
    atomic_json(manifest_path, manifest)

    for family in families:
        scenario_dir = scenario_root / family
        family_output = output_root / family
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_experiment_matrix.py"),
            "--scenario-dir", str(scenario_dir),
            "--output-dir", str(family_output),
            "--variants", args.variants,
            "--seeds", args.seeds,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--max-html-length", str(args.max_html_length),
            "--max-dom-length", str(args.max_dom_length),
            "--target-fpr", str(args.target_fpr),
            "--fpr-confidence", str(args.fpr_confidence),
            "--bootstrap-samples", str(args.bootstrap_samples),
            "--patience", str(args.patience),
            "--aux-warmup-epochs", str(args.aux_warmup_epochs),
            "--min-free-gb", str(args.min_free_gb),
            "--fusion-type", args.fusion_type,
            "--modality-dropout", str(args.modality_dropout),
            "--artifact-augmentation-probability",
            str(args.artifact_augmentation_probability),
            "--artifact-augmentation-modes", args.artifact_augmentation_modes,
            "--artifact-test-variants", args.artifact_test_variants,
        ]
        if args.with_html:
            command.append("--with-html")
        if args.freeze_html:
            command.append("--freeze-html")
        if args.artifact_tests:
            command.append("--artifact-tests")
        if args.overwrite:
            command.append("--overwrite")

        record = {
            "state": "running",
            "scenario_dir": str(scenario_dir),
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

    print(f"Multi-family LOFO experiments: {output_root}")


if __name__ == "__main__":
    main()
