# Copyright (c) 2026 Le Thanh Loi. All rights reserved.
# Author: Le Thanh Loi

"""Audit CompPhish and create a duplicate-safe, domain-disjoint split."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from compphish_common import (
    ALL_MODEL_FEATURES,
    ALL_PUBLISHED_FEATURES,
    BRAND_COLUMNS,
    FEATURE_SETS,
    HTML_COLUMNS,
    LABEL_DERIVED_COLUMNS,
    REQUIRED_RAW_COLUMNS,
    URL_COLUMNS,
    hostname_from_url,
    make_sample_id,
    normalize_url,
    project_path,
    registered_domain,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/raw/CompPhish/All_Features_threshold90.xlsx",
    )
    parser.add_argument(
        "--output",
        default="data/processed/compphish/compphish_clean_split.csv.gz",
    )
    parser.add_argument("--audit-dir", default="artifacts/compphish/data_audit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--test-folds",
        type=int,
        default=3,
        help="Number of the 20 domain-grouped folds reserved for Test (default: 3).",
    )
    return parser.parse_args()


def assign_splits(frame: pd.DataFrame, seed: int, test_folds: int) -> np.ndarray:
    """Create a Train/Test split while keeping every eTLD+1 in one set."""
    if not 1 <= test_folds <= 10:
        raise ValueError("--test-folds must be between 1 and 10")
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    fold_id = np.full(len(frame), -1, dtype=int)
    y = frame["label_binary"].to_numpy()
    groups = frame["registered_domain"].to_numpy()
    for number, (_, holdout) in enumerate(splitter.split(frame, y, groups)):
        fold_id[holdout] = number
    if (fold_id < 0).any():
        raise RuntimeError("At least one sample was not assigned to a split")
    return np.where(fold_id < 20 - test_folds, "train", "test")


def main() -> None:
    args = arguments()
    source = project_path(args.input, must_exist=True)
    destination = project_path(args.output)
    audit_dir = project_path(args.audit_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    try:
        frame = pd.read_excel(source, engine="openpyxl")
    except ImportError as error:
        raise SystemExit(
            "Missing openpyxl. Install it with: python -m pip install openpyxl"
        ) from error

    frame.columns = [str(column).strip() for column in frame.columns]
    # Version 4's workbook contains the typo ``shortner`` although the
    # accompanying extractor uses the correctly spelled ``shortener``.
    # Normalize this schema difference once so every downstream artifact uses
    # one stable, documented feature name.
    source_aliases = {
        "presence_of_url_shortner": "presence_of_url_shortener",
    }
    applicable_aliases = {
        source: target
        for source, target in source_aliases.items()
        if source in frame.columns and target not in frame.columns
    }
    frame = frame.rename(columns=applicable_aliases)
    missing = sorted(REQUIRED_RAW_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"CompPhish workbook is missing columns: {missing}")
    frame = frame[["serial_number", "url", "label", *ALL_PUBLISHED_FEATURES]].copy()
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("Expected CompPhish labels 0=legitimate and 1=phishing")

    original_rows = len(frame)
    frame["normalized_url"] = frame["url"].map(normalize_url)
    blank_urls = int(frame["normalized_url"].eq("").sum())
    if blank_urls:
        frame = frame[frame["normalized_url"].ne("")].copy()

    conflict_count = frame.groupby("normalized_url")["label"].nunique()
    conflicting_urls = set(conflict_count[conflict_count > 1].index)
    conflicts = frame[frame["normalized_url"].isin(conflicting_urls)].copy()
    conflicts.to_csv(audit_dir / "conflicting_urls.csv", index=False, encoding="utf-8-sig")
    frame = frame[~frame["normalized_url"].isin(conflicting_urls)].copy()

    duplicate_rows = int(frame.duplicated("normalized_url", keep="first").sum())
    duplicates = frame[frame.duplicated("normalized_url", keep=False)].copy()
    duplicates.to_csv(audit_dir / "duplicate_urls.csv", index=False, encoding="utf-8-sig")
    frame = frame.drop_duplicates("normalized_url", keep="first").reset_index(drop=True)

    frame = frame.rename(columns={"label": "label_binary"})
    frame["hostname"] = frame["url"].map(hostname_from_url)
    frame["registered_domain"] = frame["hostname"].map(registered_domain)
    empty_domain = frame["registered_domain"].eq("")
    frame.loc[empty_domain, "registered_domain"] = frame.loc[empty_domain, "serial_number"].map(
        lambda value: f"unknown-{value}"
    )
    frame["sample_id"] = [
        make_sample_id(url, serial)
        for url, serial in zip(frame["normalized_url"], frame["serial_number"])
    ]

    for column in ALL_PUBLISHED_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    missing_feature_values = int(frame[ALL_PUBLISHED_FEATURES].isna().sum().sum())

    frame["split"] = assign_splits(frame, args.seed, args.test_folds)
    split_domains = {
        name: set(frame.loc[frame["split"].eq(name), "registered_domain"])
        for name in ("train", "test")
    }
    overlap = {
        "train_test": len(split_domains["train"] & split_domains["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Registered-domain leakage detected: {overlap}")

    ordered = [
        "sample_id",
        "serial_number",
        "url",
        "normalized_url",
        "hostname",
        "registered_domain",
        "label_binary",
        "split",
        *ALL_PUBLISHED_FEATURES,
    ]
    frame[ordered].to_csv(
        destination, index=False, encoding="utf-8-sig", compression="gzip"
    )

    split_label = pd.crosstab(frame["split"], frame["label_binary"], margins=True)
    split_label = split_label.rename(columns={0: "legitimate", 1: "phishing"})
    split_label.to_csv(audit_dir / "split_label.csv", encoding="utf-8-sig")

    feature_rows = []
    # Descriptive association is calculated on Train only so that the Test set
    # remains untouched even during data understanding.
    audit_frame = frame[frame["split"].eq("train")]
    y = audit_frame["label_binary"].to_numpy()
    for column in ALL_PUBLISHED_FEATURES:
        values = audit_frame[column].fillna(audit_frame[column].median()).to_numpy(dtype=float)
        auc = float(roc_auc_score(y, values))
        feature_rows.append({
            "feature": column,
            "group": (
                "brand" if column in BRAND_COLUMNS
                else "url" if column in URL_COLUMNS or column in LABEL_DERIVED_COLUMNS
                else "html"
            ),
            "used_for_modeling": column in ALL_MODEL_FEATURES,
            "raw_auc": auc,
            "direction_free_auc": max(auc, 1.0 - auc),
            "direction": "higher_phishing" if auc >= 0.5 else "lower_phishing",
        })
    pd.DataFrame(feature_rows).sort_values(
        "direction_free_auc", ascending=False
    ).to_csv(audit_dir / "univariate_feature_audit.csv", index=False, encoding="utf-8-sig")

    summary = {
        "dataset": "CompPhish v4",
        "original_rows": original_rows,
        "clean_rows": len(frame),
        "blank_url_rows_removed": blank_urls,
        "conflicting_normalized_url_rows_removed": int(len(conflicts)),
        "duplicate_normalized_url_rows_removed": duplicate_rows,
        "unique_registered_domains": int(frame["registered_domain"].nunique()),
        "label_counts": {
            "legitimate": int(frame["label_binary"].eq(0).sum()),
            "phishing": int(frame["label_binary"].eq(1).sum()),
        },
        "split_counts": frame["split"].value_counts().to_dict(),
        "split_label_counts": {
            split_name: {
                "legitimate": int(frame.loc[frame["split"].eq(split_name), "label_binary"].eq(0).sum()),
                "phishing": int(frame.loc[frame["split"].eq(split_name), "label_binary"].eq(1).sum()),
            }
            for split_name in ("train", "test")
        },
        "domain_overlap": overlap,
        "missing_feature_values": missing_feature_values,
        "published_feature_count": len(ALL_PUBLISHED_FEATURES),
        "model_feature_count": len(ALL_MODEL_FEATURES),
        "excluded_label_derived_features": LABEL_DERIVED_COLUMNS,
        "feature_sets": {name: len(columns) for name, columns in FEATURE_SETS.items()},
        "split_method": (
            "20-fold StratifiedGroupKFold grouped by registered domain; "
            f"{20 - args.test_folds}/{args.test_folds} folds assigned to Train/Test"
        ),
        "seed": args.seed,
    }
    (audit_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved processed data: {destination}")
    print(f"Saved audit: {audit_dir}")


if __name__ == "__main__":
    main()
