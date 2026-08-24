import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research import environment_metadata, git_metadata, sha256_file, source_snapshot

VARIANTS = {
    "bce": (0.0, 0.0, 0.0, 0.0),
    "supcon": (0.05, 0.0, 0.0, 0.0),
    "supcon_adv": (0.05, 0.02, 0.0, 0.0),
    "supcon_coral": (0.05, 0.0, 0.2, 0.0),
    "full": (0.05, 0.02, 0.2, 0.0),
    "artifact_consistency": (0.0, 0.0, 0.0, 0.1),
    "robust_full": (0.05, 0.02, 0.2, 0.1),
}


def resolve_split(scenario_dir, name):
    manifest_path = scenario_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        declared = manifest.get("files", {}).get(name)
        if declared and (scenario_dir / declared).exists():
            return scenario_dir / declared
    for filename in (f"{name}.csv.gz", f"{name}.csv"):
        candidate = scenario_dir / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No {name}.csv or {name}.csv.gz in {scenario_dir}")


def run_command(command, dry_run=False):
    print("\nRUN:", subprocess.list2cmdline([str(value) for value in command]), flush=True)
    if not dry_run:
        subprocess.run([str(value) for value in command], cwd=ROOT, check=True)


def require_disk(path, minimum_gb):
    path.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / (1024 ** 3)
    print(f"Free disk: {free_gb:.2f} GB", flush=True)
    if free_gb < minimum_gb:
        raise RuntimeError(
            f"Stopping with {free_gb:.2f} GB free; minimum is {minimum_gb:.2f} GB"
        )


def write_manifest(path, manifest):
    """Atomically persist run state so interrupted matrices remain auditable."""
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(
        description="Run storage-aware GIPRL ablations and seeds with calibrated evaluation."
    )
    parser.add_argument("--scenario-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--variants", default="bce,supcon,supcon_adv,supcon_coral,full")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-html-length", type=int, default=256)
    parser.add_argument("--max-dom-length", type=int, default=256)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--fpr-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--aux-warmup-epochs", type=float, default=1.0)
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    parser.add_argument("--with-html", action="store_true")
    parser.add_argument("--freeze-html", action="store_true")
    parser.add_argument("--fusion-type", choices=["concat", "gated"], default="gated")
    parser.add_argument("--modality-dropout", type=float, default=0.1)
    parser.add_argument("--artifact-augmentation-probability", type=float, default=0.3)
    parser.add_argument(
        "--artifact-augmentation-modes",
        default="comments_removed,identifiers_normalized,visible_text_removed",
    )
    parser.add_argument("--artifact-tests", action="store_true")
    parser.add_argument(
        "--artifact-test-variants", default="supcon,full,robust_full",
        help="Comma-separated variants that receive expensive artifact stress tests",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scenario_dir = Path(args.scenario_dir).resolve()
    train = resolve_split(scenario_dir, "train")
    val = resolve_split(scenario_dir, "val")
    test = resolve_split(scenario_dir, "test")
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    artifact_test_variants = {
        value.strip() for value in args.artifact_test_variants.split(",") if value.strip()
    }
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}; choices={sorted(VARIANTS)}")
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")

    output_root = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (ROOT / "outputs" / "experiments" / scenario_dir.name)
    )
    require_disk(output_root, args.min_free_gb)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenario_dir": str(scenario_dir),
        "output_dir": str(output_root),
        "variants": variants,
        "seeds": seeds,
        "with_html": args.with_html,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "target_fpr": args.target_fpr,
        "fpr_confidence": args.fpr_confidence,
        "artifact_augmentation_probability": args.artifact_augmentation_probability,
        "artifact_augmentation_modes": args.artifact_augmentation_modes,
        "git": git_metadata(ROOT),
        "source_snapshot": source_snapshot(ROOT),
        "environment": environment_metadata(),
        "input_sha256": {
            "train": sha256_file(train),
            "val": sha256_file(val),
            "test": sha256_file(test),
        },
        "status": {},
    }
    manifest_path = output_root / "run_manifest.json"
    if not args.dry_run:
        write_manifest(manifest_path, manifest)

    for variant in variants:
        (
            lambda_supcon,
            lambda_adversarial,
            lambda_alignment,
            lambda_artifact_consistency,
        ) = VARIANTS[variant]
        manifest["status"][variant] = {}
        for seed in seeds:
            require_disk(output_root, args.min_free_gb)
            run_dir = output_root / variant / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = run_dir / "best_model.pt"
            metrics = run_dir / "metrics.json"
            predictions = run_dir / "predictions.csv"
            run_status = {
                "state": "pending",
                "checkpoint": str(checkpoint),
                "metrics": str(metrics),
                "predictions": str(predictions),
            }
            manifest["status"][variant][str(seed)] = run_status
            if not args.dry_run:
                write_manifest(manifest_path, manifest)

            train_command = [
                sys.executable, "-u", ROOT / "scripts" / "train_deep.py",
                "--train", train, "--val", val,
                "--epochs", args.epochs, "--batch-size", args.batch_size,
                "--max-html-length", args.max_html_length,
                "--max-dom-length", args.max_dom_length,
                "--lambda-supcon", lambda_supcon,
                "--lambda-adversarial", lambda_adversarial,
                "--lambda-alignment", lambda_alignment,
                "--lambda-artifact-consistency", lambda_artifact_consistency,
                "--seed", seed, "--patience", args.patience,
                "--aux-warmup-epochs", args.aux_warmup_epochs,
                "--fusion-type", args.fusion_type,
                "--modality-dropout", args.modality_dropout,
                "--artifact-augmentation-probability", args.artifact_augmentation_probability,
                "--artifact-augmentation-modes", args.artifact_augmentation_modes,
                "--min-free-gb", args.min_free_gb,
                "--output", checkpoint,
            ]
            if not args.with_html:
                train_command.append("--disable-html")
            if args.freeze_html:
                train_command.append("--freeze-html")
            if args.overwrite:
                train_command.append("--overwrite")

            try:
                run_status["state"] = "training"
                if not args.dry_run:
                    write_manifest(manifest_path, manifest)
                if args.overwrite or not checkpoint.exists():
                    run_command(train_command, dry_run=args.dry_run)
                else:
                    print(f"SKIP existing checkpoint: {checkpoint}")
                if not args.dry_run:
                    run_status["checkpoint_sha256"] = sha256_file(checkpoint)

                evaluate_command = [
                    sys.executable, "-u", ROOT / "scripts" / "evaluate_deep.py",
                    "--checkpoint", checkpoint, "--val", val, "--test", test,
                    "--batch-size", args.batch_size, "--target-fpr", args.target_fpr,
                    "--fpr-confidence", args.fpr_confidence,
                    "--output-json", metrics, "--predictions", predictions,
                ]
                if args.artifact_tests and variant in artifact_test_variants:
                    evaluate_command.append("--artifact-tests")
                run_status["state"] = "evaluating"
                if not args.dry_run:
                    write_manifest(manifest_path, manifest)
                if args.overwrite or not (metrics.exists() and predictions.exists()):
                    run_command(evaluate_command, dry_run=args.dry_run)
                else:
                    print(f"SKIP existing metrics: {metrics}")
                if not args.dry_run:
                    run_status.update({
                        "state": "complete",
                        "metrics_sha256": sha256_file(metrics),
                        "predictions_sha256": sha256_file(predictions),
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    })
                    write_manifest(manifest_path, manifest)
            except Exception as exc:
                run_status.update({
                    "state": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                })
                if not args.dry_run:
                    write_manifest(manifest_path, manifest)
                raise

    summary_command = [
        sys.executable, "-u", ROOT / "scripts" / "summarize_experiments.py",
        "--root", output_root,
        "--out-json", output_root / "summary.json",
        "--out-csv", output_root / "summary.csv",
        "--bootstrap-samples", args.bootstrap_samples,
    ]
    run_command(summary_command, dry_run=args.dry_run)
    print(f"\nExperiment matrix complete: {output_root}")


if __name__ == "__main__":
    main()
