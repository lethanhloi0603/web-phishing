# Copyright (c) 2026 Le Thanh Loi. All rights reserved.
# Author: Le Thanh Loi

"""Create local TreeSHAP explanations for representative TP, FN and FP cases.

The fitted Full CatBoost pipeline and the existing integrity-clean Test split
are reused.  No model is trained or updated by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compphish_common import FEATURE_SETS_PUBLISHED, project_path, read_processed


CASE_DEFINITIONS = {
    "tp": {
        "label": "Phishing được dự đoán đúng",
        "true_label": 1,
        "predicted_label": 1,
        "filename": "figure_local_tp.png",
    },
    "fn": {
        "label": "Phishing bị bỏ sót",
        "true_label": 1,
        "predicted_label": 0,
        "filename": "figure_local_fn.png",
    },
    "fp": {
        "label": "Website hợp lệ bị cảnh báo nhầm",
        "true_label": 0,
        "predicted_label": 1,
        "filename": "figure_local_fp.png",
    },
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/processed/compphish/compphish_integrity_split.csv.gz",
    )
    parser.add_argument(
        "--experiment-dir",
        default="artifacts/compphish/branch_stacking_full",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/compphish/branch_stacking_xai_70/local_cases",
    )
    parser.add_argument("--feature-model", default="full__catboost")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-display", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def positive_class_shap(values: object, feature_count: int) -> np.ndarray:
    """Normalize version-dependent SHAP output for binary classification."""
    if isinstance(values, list):
        array = np.asarray(values[-1])
    else:
        array = np.asarray(values)
    if array.ndim == 3:
        if array.shape[-1] == 2:
            array = array[:, :, 1]
        elif array.shape[0] == 2:
            array = array[1]
    if array.ndim != 2 or array.shape[1] != feature_count:
        raise ValueError(f"Unexpected SHAP shape: {array.shape}")
    return array


def positive_class_base_value(value: object) -> float:
    """Return the positive-class TreeSHAP expected value."""
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError("TreeSHAP returned no expected value")
    return float(array[-1])


def model_feature_columns(experiment: Path, model_name: str) -> list[str]:
    members_path = experiment / "base_members.csv"
    if members_path.exists():
        members = pd.read_csv(members_path, dtype=str).fillna("")
        match = members[members["model"].eq(model_name)]
        if not match.empty and match.iloc[0]["input_features"]:
            return match.iloc[0]["input_features"].split("|")
    return list(FEATURE_SETS_PUBLISHED["full"])


def explained_feature_names(branch: object, feature_columns: list[str]) -> list[str]:
    selector = branch.named_steps.get("selector")
    if selector is not None and selector != "passthrough" and hasattr(selector, "get_support"):
        return [
            feature
            for feature, keep in zip(feature_columns, selector.get_support())
            if keep
        ]
    return list(feature_columns)


def representative_position(
    table: pd.DataFrame, true_label: int, predicted_label: int,
) -> int:
    """Select the score-nearest-to-median row inside one error category."""
    candidates = table[
        table["label_binary"].eq(true_label)
        & table["predicted_label"].eq(predicted_label)
    ].copy()
    if candidates.empty:
        raise ValueError(
            "No Test observation matches true_label="
            f"{true_label}, predicted_label={predicted_label}"
        )
    score_median = float(candidates["phishing_score"].median())
    distance = (candidates["phishing_score"] - score_median).abs()
    # mergesort is stable, so ties are resolved reproducibly by Test order.
    return int(distance.sort_values(kind="mergesort").index[0])


def waterfall_figure(
    shap_module: object,
    contribution: np.ndarray,
    base_value: float,
    raw_values: np.ndarray,
    feature_names: list[str],
    title: str,
    output_path: Path,
    max_display: int,
    dpi: int,
) -> None:
    explanation = shap_module.Explanation(
        values=contribution,
        base_values=base_value,
        # Contributions are computed from the transformed pipeline input, but
        # the displayed values remain on the published feature scale so the
        # figure is interpretable in the thesis.
        data=raw_values,
        feature_names=feature_names,
    )
    plt.figure(figsize=(10.5, 6.8))
    shap_module.plots.waterfall(
        explanation, max_display=max_display, show=False,
    )
    figure = plt.gcf()
    figure.set_size_inches(10.5, 6.8)
    plt.title(title, pad=18, fontsize=12, weight="bold")
    plt.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = arguments()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.max_display < 2:
        raise ValueError("--max-display must be at least 2")

    try:
        import shap
    except ImportError as error:
        raise SystemExit("Install SHAP with: python -m pip install shap") from error

    experiment = project_path(args.experiment_dir, must_exist=True)
    output = project_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    model_path = experiment / "models" / f"{args.feature_model}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing fitted model: {model_path}. This script does not retrain it."
        )
    branch = joblib.load(model_path)

    frame = read_processed(args.data)
    test = frame[frame["split"].eq("test")].reset_index(drop=True)
    if test.empty:
        raise ValueError("The processed dataset contains no Test rows")

    feature_columns = model_feature_columns(experiment, args.feature_model)
    missing = sorted(set(feature_columns) - set(test.columns))
    if missing:
        raise ValueError(f"Test data is missing model features: {missing}")

    score = branch.predict_proba(test[feature_columns])[:, 1]
    predicted_label = (score >= args.threshold).astype(int)
    cases = test.copy()
    cases["phishing_score"] = score
    cases["predicted_label"] = predicted_label

    selected_positions = {
        case_type: representative_position(
            cases,
            int(definition["true_label"]),
            int(definition["predicted_label"]),
        )
        for case_type, definition in CASE_DEFINITIONS.items()
    }
    selected = test.iloc[list(selected_positions.values())].copy()

    transformed = branch[:-1].transform(selected[feature_columns])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed, dtype=float)
    explained_features = explained_feature_names(branch, feature_columns)
    if transformed.shape[1] != len(explained_features):
        raise ValueError(
            "Transformed feature count does not match explained feature names: "
            f"{transformed.shape[1]} versus {len(explained_features)}"
        )

    estimator = branch.named_steps["model"]
    explainer = shap.TreeExplainer(estimator)
    shap_values = positive_class_shap(
        explainer.shap_values(transformed), len(explained_features)
    )
    base_value = positive_class_base_value(explainer.expected_value)

    summary_rows: list[dict[str, object]] = []
    contribution_rows: list[dict[str, object]] = []
    for selected_number, (case_type, test_position) in enumerate(
        selected_positions.items()
    ):
        definition = CASE_DEFINITIONS[case_type]
        test_row = cases.iloc[test_position]
        contribution = shap_values[selected_number]
        raw_values = pd.to_numeric(
            selected.iloc[selected_number][explained_features], errors="coerce"
        ).to_numpy(dtype=float)

        summary_rows.append({
            "case_type": case_type,
            "case_label_vi": definition["label"],
            "selection_rule": "score nearest to category median",
            "sample_id": test_row.get("sample_id", ""),
            "serial_number": test_row.get("serial_number", ""),
            "url": test_row.get("url", ""),
            "registered_domain": test_row.get("registered_domain", ""),
            "true_label": int(test_row["label_binary"]),
            "predicted_label": int(test_row["predicted_label"]),
            "phishing_probability": float(test_row["phishing_score"]),
            "threshold": args.threshold,
        })

        ranking = np.argsort(np.abs(contribution))[::-1]
        for rank, feature_index in enumerate(ranking, start=1):
            contribution_rows.append({
                "case_type": case_type,
                "sample_id": test_row.get("sample_id", ""),
                "rank": rank,
                "feature": explained_features[feature_index],
                "raw_feature_value": raw_values[feature_index],
                "transformed_feature_value": transformed[
                    selected_number, feature_index
                ],
                "shap_value": float(contribution[feature_index]),
                "absolute_shap_value": float(abs(contribution[feature_index])),
                "direction": (
                    "toward_phishing"
                    if contribution[feature_index] > 0
                    else "toward_legitimate"
                ),
            })

        waterfall_figure(
            shap_module=shap,
            contribution=contribution,
            base_value=base_value,
            raw_values=raw_values,
            feature_names=explained_features,
            title=(
                f"{definition['label']} | "
                f"P(phishing)={float(test_row['phishing_score']):.4f}"
            ),
            output_path=output / str(definition["filename"]),
            max_display=args.max_display,
            dpi=args.dpi,
        )

    pd.DataFrame(summary_rows).to_csv(
        output / "local_case_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(contribution_rows).to_csv(
        output / "local_case_contributions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    scope = {
        "feature_model": args.feature_model,
        "model_path": str(model_path),
        "dataset": "CompPhish v4 integrity-clean Test split",
        "test_rows": len(test),
        "feature_count": len(explained_features),
        "threshold": args.threshold,
        "case_types": list(CASE_DEFINITIONS),
        "selection_rule": (
            "Within each TP/FN/FP category, select the observation whose "
            "phishing score is nearest to that category's median score."
        ),
        "shap_scope": (
            "Local TreeSHAP explanations of Full CatBoost only; not an "
            "explanation of either stacking ensemble."
        ),
        "causal_warning": (
            "SHAP values explain model-output contributions and do not establish "
            "causal relationships in real phishing behavior."
        ),
        "display_note": (
            "SHAP contributions use transformed pipeline inputs; waterfall "
            "feature values are displayed on the original published scale."
        ),
    }
    (output / "local_xai_scope.json").write_text(
        json.dumps(scope, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Selected local XAI cases:")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\nSaved local XAI outputs: {output}")


if __name__ == "__main__":
    main()
