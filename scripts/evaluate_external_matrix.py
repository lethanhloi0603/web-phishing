import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import read_frame
from src.research import evaluation_data_evidence, sha256_file


def require_disk(path, minimum_gb):
    path.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / 2**30
    if free_gb < minimum_gb:
        raise RuntimeError(
            f"Stopping with {free_gb:.2f} GB free; minimum is {minimum_gb:.2f} GB"
        )
    return free_gb


def validate_external_frame(frame, path, min_sites):
    evidence = evaluation_data_evidence(frame, path)
    problems = []
    for label, field in (("phishing", "phishing_sites"), ("benign", "benign_sites")):
        if int(evidence.get(field, 0)) < min_sites:
            problems.append(f"{field}={evidence.get(field, 0)}, required>={min_sites}")
    if int(evidence.get("site_label_conflicts", 0)):
        problems.append(
            f"site_label_conflicts={evidence['site_label_conflicts']}, required=0"
        )
    provenance = {
        str(value).strip().lower()
        for value in evidence.get("dataset_provenance", [])
        if str(value).strip().lower() not in {"", "unknown", "none", "nan"}
    }
    if not provenance:
        problems.append("dataset_provenance is missing")
    if problems:
        raise ValueError("External test is not site-valid: " + "; ".join(problems))
    return evidence


def evaluation_command(
    checkpoint, val, test, metrics, predictions, batch_size,
    target_fpr, fpr_confidence, artifact_tests=False,
):
    command = [
        sys.executable, "-u", ROOT / "scripts" / "evaluate_deep.py",
        "--checkpoint", checkpoint,
        "--val", val,
        "--test", test,
        "--batch-size", batch_size,
        "--target-fpr", target_fpr,
        "--fpr-confidence", fpr_confidence,
        "--evaluation-role", "external",
        "--output-json", metrics,
        "--predictions", predictions,
    ]
    if artifact_tests:
        command.append("--artifact-tests")
    return [str(value) for value in command]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the exact primary multi-seed checkpoints on one external test set."
    )
    parser.add_argument("--checkpoints-root", required=True)
    parser.add_argument("--variant", default="robust_full")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--val", required=True, help="Primary validation split for thresholds")
    parser.add_argument("--test", required=True, help="Independent normalized external CSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--fpr-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--min-external-sites-per-class", type=int, default=100)
    parser.add_argument("--min-free-gb", type=float, default=6.0)
    parser.add_argument("--artifact-tests", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    checkpoints_root = Path(args.checkpoints_root).resolve()
    val = Path(args.val).resolve()
    test = Path(args.test).resolve()
    output_root = Path(args.output_dir).resolve()
    require_disk(output_root, args.min_free_gb)
    frame = read_frame(test)
    external_evidence = validate_external_frame(
        frame, test, args.min_external_sites_per_class
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_role": "external",
        "checkpoints_root": str(checkpoints_root),
        "variant": args.variant,
        "seeds": seeds,
        "validation_sha256": sha256_file(val),
        "external_evidence": external_evidence,
        "runs": {},
    }
    for seed in seeds:
        require_disk(output_root, args.min_free_gb)
        checkpoint = checkpoints_root / args.variant / f"seed_{seed}" / "best_model.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing primary checkpoint: {checkpoint}")
        run_dir = output_root / args.variant / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics = run_dir / "metrics.json"
        predictions = run_dir / "predictions.csv"
        if args.overwrite or not (metrics.exists() and predictions.exists()):
            command = evaluation_command(
                checkpoint, val, test, metrics, predictions,
                args.batch_size, args.target_fpr, args.fpr_confidence,
                artifact_tests=args.artifact_tests,
            )
            print("RUN:", subprocess.list2cmdline(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
        else:
            print(f"SKIP existing external metrics: {metrics}")
        manifest["runs"][str(seed)] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "metrics": str(metrics),
            "predictions": str(predictions),
        }
        (output_root / "evaluation_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    summary_command = [
        sys.executable, "-u", ROOT / "scripts" / "summarize_experiments.py",
        "--root", output_root,
        "--out-json", output_root / "summary.json",
        "--out-csv", output_root / "summary.csv",
        "--bootstrap-samples", args.bootstrap_samples,
    ]
    subprocess.run([str(value) for value in summary_command], cwd=ROOT, check=True)
    print(f"External matrix complete: {output_root}")


if __name__ == "__main__":
    main()
