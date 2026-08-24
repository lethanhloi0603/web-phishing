import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score


def main():
    parser = argparse.ArgumentParser(description="Measure agreement between two human annotators")
    parser.add_argument("--audit", required=True)
    parser.add_argument("--reviewer-a", required=True, help="Column containing reviewer A labels")
    parser.add_argument("--reviewer-b", required=True, help="Column containing reviewer B labels")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    frame = pd.read_csv(args.audit)
    missing = {args.reviewer_a, args.reviewer_b} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing reviewer columns: {sorted(missing)}")
    paired = frame[[args.reviewer_a, args.reviewer_b]].dropna().astype(str)
    if paired.empty:
        raise ValueError("No samples labeled by both reviewers")
    agreement = float((paired[args.reviewer_a] == paired[args.reviewer_b]).mean())
    labels = sorted(set(paired[args.reviewer_a]) | set(paired[args.reviewer_b]))
    matrix = pd.crosstab(paired[args.reviewer_a], paired[args.reviewer_b], dropna=False)
    report = {
        "n_double_annotated": int(len(paired)),
        "labels": labels,
        "raw_agreement": agreement,
        "cohen_kappa": float(cohen_kappa_score(paired[args.reviewer_a], paired[args.reviewer_b], labels=labels)),
        "confusion_matrix": {
            str(row): {str(column): int(matrix.loc[row, column]) for column in matrix.columns}
            for row in matrix.index
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
