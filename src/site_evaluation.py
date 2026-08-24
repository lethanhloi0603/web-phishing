import numpy as np
import pandas as pd

from .calibration import calibrated_evaluation
from .metrics import binary_metrics


def aggregate_site_predictions(frame, aggregation="max"):
    if aggregation not in {"max", "mean", "noisy_or"}:
        raise ValueError("aggregation must be max, mean, or noisy_or")
    if "probability" not in frame:
        raise ValueError("frame must contain probability")

    data = frame.copy()
    if "codebase_id" not in data:
        data["codebase_id"] = (
            data["sample_id"].astype(str)
            if "sample_id" in data
            else [f"row-{index}" for index in range(len(data))]
        )
    data["codebase_id"] = data["codebase_id"].fillna("").astype(str)
    empty = data["codebase_id"].str.strip().eq("")
    fallback = data["sample_id"].astype(str) if "sample_id" in data else data.index.astype(str)
    data.loc[empty, "codebase_id"] = fallback[empty]

    label_column = "site_label" if "site_label" in data else "label"
    conflicts = data.groupby("codebase_id")[label_column].nunique(dropna=False)
    if conflicts.gt(1).any():
        examples = conflicts[conflicts.gt(1)].index.astype(str).tolist()[:5]
        raise ValueError(f"Conflicting site labels within codebases: {examples}")

    def aggregate_probability(values):
        values = np.asarray(values, dtype=float)
        if aggregation == "max":
            return float(values.max())
        if aggregation == "mean":
            return float(values.mean())
        return float(1.0 - np.prod(1.0 - np.clip(values, 0.0, 1.0)))

    metadata = [
        column for column in (
            "generator", "generator_family", "brand", "dataset_domain",
            "dataset_provenance", "label_provenance",
        ) if column in data
    ]
    rows = []
    for codebase_id, group in data.groupby("codebase_id", sort=False):
        row = {
            "codebase_id": codebase_id,
            "label": int(pd.to_numeric(group[label_column]).iloc[0]),
            "probability": aggregate_probability(group["probability"]),
            "n_pages": int(len(group)),
            "positive_page_labels": int(pd.to_numeric(group["label"]).eq(1).sum()),
        }
        for column in metadata:
            row[column] = str(group[column].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def site_level_evaluation(
    val_predictions,
    test_predictions,
    target_fpr=0.01,
    aggregation="max",
    fpr_confidence=0.95,
):
    val_sites = aggregate_site_predictions(val_predictions, aggregation=aggregation)
    test_sites = aggregate_site_predictions(test_predictions, aggregation=aggregation)
    report = {
        "aggregation": aggregation,
        "validation_sites": int(len(val_sites)),
        "test_sites": int(len(test_sites)),
        "overall": binary_metrics(test_sites["label"], test_sites["probability"]),
        "calibration": calibrated_evaluation(
            val_sites["label"].to_numpy(), val_sites["probability"].to_numpy(),
            test_sites["label"].to_numpy(), test_sites["probability"].to_numpy(),
            target_fpr=target_fpr,
            fpr_confidence=fpr_confidence,
        ),
    }
    return report, val_sites, test_sites
