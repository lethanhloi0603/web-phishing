# Copyright (c) 2026 Le Thanh Loi. All rights reserved.
# Author: Le Thanh Loi

"""Run reproducible CompPhish shortcut and feature-selection ablations.

Each variant delegates to train_compphish_models.py and therefore uses the
same domain-grouped Train OOF protocol. The script aggregates results but does
not choose a winning variant from Test performance.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import pandas as pd

from compphish_common import project_path


VARIANTS = {
    "baseline": [],
    "drop_count_www": ["--drop-features", "count_www"],
    "drop_visible_url_cues": ["--drop-visible-url-cues"],
    "mutual_info": ["--feature-selector", "mutual_info"],
    "l1": ["--feature-selector", "l1"],
    "tree": ["--feature-selector", "tree"],
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="data/processed/compphish/compphish_clean_split.csv.gz"
    )
    parser.add_argument(
        "--output-dir", default="artifacts/compphish/ablation_research"
    )
    parser.add_argument(
        "--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS)
    )
    parser.add_argument("--views", nargs="+", default=["full"])
    parser.add_argument("--model-set", choices=("core", "extended"), default="core")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--optional-boosters", choices=("auto", "none", "required"), default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--train-limit", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    output = project_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trainer = project_path("scripts/train_compphish_models.py", must_exist=True)
    rows = []

    for variant in args.variants:
        destination = output / variant
        command = [
            sys.executable,
            str(trainer),
            "--data", str(project_path(args.data, must_exist=True)),
            "--output-dir", str(destination),
            "--cv-folds", str(args.cv_folds),
            "--model-set", args.model_set,
            "--optional-boosters", args.optional_boosters,
            "--views", *args.views,
            "--seed", str(args.seed),
            *VARIANTS[variant],
        ]
        if args.quick:
            command.extend(["--quick", "--train-limit", str(args.train_limit)])
        print(f"\n=== Ablation: {variant} ===")
        subprocess.run(command, check=True)

        metrics = pd.read_csv(destination / "metrics.csv")
        selected = json.loads(
            (destination / "selected_model.json").read_text(encoding="utf-8")
        )
        chosen = metrics[metrics["model"].eq(selected["model"])].copy()
        for record in chosen.to_dict("records"):
            record["variant"] = variant
            rows.append(record)

    summary = pd.DataFrame(rows)
    order = [
        "variant", "split", "model", "rows", "accuracy", "precision",
        "recall", "specificity", "f1", "pr_auc", "roc_auc", "mcc",
        "fpr", "fnr", "threshold_selected_on_train_oof",
    ]
    available = [column for column in order if column in summary.columns]
    summary[available].to_csv(
        output / "ablation_summary.csv", index=False, encoding="utf-8-sig"
    )
    print("\nAblation summary (descriptive; Test is not a selection set):")
    print(summary[available].to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
