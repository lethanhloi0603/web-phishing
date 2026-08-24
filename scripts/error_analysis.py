import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Aggregate repeatable false-positive/negative cases")
    parser.add_argument("--variant-dir", required=True)
    parser.add_argument(
        "--evaluation",
        choices=[
            "predicted_0_5", "predicted_target_fpr",
            "predicted_target_fpr_confident",
        ],
        default="predicted_target_fpr_confident",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    runs = []
    for path in sorted(Path(args.variant_dir).glob("seed_*/predictions.csv")):
        frame = pd.read_csv(path)
        if args.evaluation not in frame:
            raise ValueError(f"{path}: missing {args.evaluation}")
        metadata = [column for column in ("sample_id", "label", "generator", "generator_family", "brand", "template_id", "file_path") if column in frame]
        run = frame[metadata + ["probability", args.evaluation]].copy()
        run = run.rename(columns={"probability": f"probability_{path.parent.name}", args.evaluation: f"prediction_{path.parent.name}"})
        runs.append((path.parent.name, run))
    if not runs:
        raise FileNotFoundError("No seed_*/predictions.csv files found")
    key = "sample_id" if "sample_id" in runs[0][1] else None
    combined = runs[0][1]
    for seed_name, run in runs[1:]:
        shared = [key] if key else ["label"]
        selected = shared + [column for column in run if column.startswith(("probability_", "prediction_"))]
        if key:
            selected.append("label")
            run = run[selected].rename(columns={"label": f"label_{seed_name}"})
        else:
            run = run[selected]
        combined = combined.merge(run, on=shared, how="inner", validate="one_to_one" if key else None)
        if key and not combined["label"].equals(combined[f"label_{seed_name}"]):
            raise ValueError(f"Label mismatch for {seed_name}")
    probability_columns = [column for column in combined if column.startswith("probability_seed_")]
    prediction_columns = [column for column in combined if column.startswith("prediction_seed_")]
    combined["mean_probability"] = combined[probability_columns].mean(axis=1)
    combined["positive_votes"] = combined[prediction_columns].sum(axis=1)
    combined["error_type"] = "correct"
    combined.loc[(combined["label"] == 0) & (combined["positive_votes"] > 0), "error_type"] = "false_positive"
    combined.loc[(combined["label"] == 1) & (combined["positive_votes"] < len(prediction_columns)), "error_type"] = "false_negative"
    errors = combined[combined["error_type"] != "correct"].copy()
    errors["error_consistency"] = errors.apply(
        lambda row: row["positive_votes"] if row["error_type"] == "false_positive" else len(prediction_columns) - row["positive_votes"], axis=1
    )
    errors = errors.sort_values(["error_consistency", "mean_probability"], ascending=[False, False])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(out, index=False)
    print(f"Exported {len(errors)} error cases to {out}")


if __name__ == "__main__":
    main()
