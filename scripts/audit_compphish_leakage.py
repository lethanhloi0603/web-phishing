# Copyright (c) 2026 Le Thanh Loi. All rights reserved.
# Author: Le Thanh Loi

"""Audit cross-Train/Test leakage in CompPhish features and raw HTML.

Checks are descriptive only and never alter the processed dataset:
1. exact duplicate 70-feature vectors across Train and Test.  The two
   label-conditioned mean indicators are recomputed from Train only;
2. nearest standardized Train vector for every Test row;
3. exact raw and whitespace-normalized HTML duplicates across sets;
4. near-duplicate HTML candidates using 64-bit SimHash and four-band LSH.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from compphish_common import (
    ALL_PUBLISHED_FEATURES,
    FoldSafePhishingMeanFeatures,
    project_path,
    read_processed,
)


TOKEN_PATTERN = re.compile(r"[a-z0-9_./:@?=&%+-]+", flags=re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(rb"\s+")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="data/processed/compphish/compphish_integrity_split.csv.gz"
    )
    parser.add_argument("--html-zip", default="data/raw/CompPhish/All_HTML.zip")
    parser.add_argument(
        "--output-dir", default="artifacts/compphish/leakage_audit_integrity_70"
    )
    parser.add_argument("--near-html-distance", type=int, default=3)
    parser.add_argument("--nearest-feature-rows", type=int, default=250)
    parser.add_argument("--skip-html", action="store_true")
    return parser.parse_args()


def normalize_html_bytes(content: bytes) -> bytes:
    return WHITESPACE_PATTERN.sub(b" ", content.strip().lower())


def simhash64(content: bytes) -> int:
    text = content.decode("utf-8", errors="ignore").lower()
    tokens = TOKEN_PATTERN.findall(text)
    if not tokens:
        tokens = [text[:4096] or "<empty>"]
    accumulator = [0] * 64
    for token, weight in Counter(tokens).items():
        value = int.from_bytes(
            hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big"
        )
        for bit in range(64):
            accumulator[bit] += weight if value & (1 << bit) else -weight
    result = 0
    for bit, value in enumerate(accumulator):
        if value >= 0:
            result |= 1 << bit
    return result


def feature_audit(
    frame: pd.DataFrame, output, nearest_feature_rows: int
) -> dict[str, object]:
    # Match the final 70-feature modelling protocol.  Never consume the two
    # workbook-wide mean flags directly: learn their thresholds from Train
    # phishing rows and apply those thresholds to both Train and Test.
    values = frame[ALL_PUBLISHED_FEATURES].apply(pd.to_numeric, errors="coerce")
    train_mask_frame = ~frame["split"].eq("test")
    fold_safe = FoldSafePhishingMeanFeatures()
    fold_safe.fit(
        values.loc[train_mask_frame],
        frame.loc[train_mask_frame, "label_binary"],
    )
    values = fold_safe.transform(values)
    hashes = pd.util.hash_pandas_object(values, index=False).astype("uint64")
    feature_hash = frame[[
        "sample_id", "serial_number", "registered_domain", "label_binary", "split"
    ]].copy()
    feature_hash["analysis_split"] = np.where(
        feature_hash["split"].eq("test"), "test", "train"
    )
    feature_hash["feature_hash"] = hashes.astype(str)
    cross_hashes = feature_hash.groupby("feature_hash")["analysis_split"].nunique()
    cross_hashes = set(cross_hashes[cross_hashes > 1].index)
    exact = feature_hash[feature_hash["feature_hash"].isin(cross_hashes)].copy()
    exact.to_csv(
        output / "exact_feature_vector_cross_split.csv",
        index=False,
        encoding="utf-8-sig",
    )

    train_mask = feature_hash["analysis_split"].eq("train").to_numpy()
    test_mask = ~train_mask
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_values = imputer.fit_transform(values.loc[train_mask])
    test_values = imputer.transform(values.loc[test_mask])
    train_values = scaler.fit_transform(train_values)
    test_values = scaler.transform(test_values)
    nearest = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=-1)
    nearest.fit(train_values)
    distances, indices = nearest.kneighbors(test_values)
    train_meta = feature_hash.loc[train_mask].reset_index(drop=True)
    test_meta = feature_hash.loc[test_mask].reset_index(drop=True)
    nearest_table = pd.DataFrame({
        "test_sample_id": test_meta["sample_id"],
        "test_serial_number": test_meta["serial_number"],
        "test_label": test_meta["label_binary"],
        "nearest_train_sample_id": train_meta.iloc[indices[:, 0]]["sample_id"].to_numpy(),
        "nearest_train_serial_number": train_meta.iloc[indices[:, 0]]["serial_number"].to_numpy(),
        "nearest_train_label": train_meta.iloc[indices[:, 0]]["label_binary"].to_numpy(),
        "standardized_euclidean_distance": distances[:, 0],
    }).sort_values("standardized_euclidean_distance")
    nearest_table.head(max(1, int(nearest_feature_rows))).to_csv(
        output / "nearest_feature_vectors.csv", index=False, encoding="utf-8-sig"
    )
    quantiles = nearest_table["standardized_euclidean_distance"].quantile(
        [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]
    )
    return {
        "cross_split_exact_feature_hashes": len(cross_hashes),
        "cross_split_exact_feature_rows": len(exact),
        "nearest_feature_distance_quantiles": {
            str(index): float(value) for index, value in quantiles.items()
        },
        "zero_distance_test_rows": int(
            np.isclose(nearest_table["standardized_euclidean_distance"], 0.0).sum()
        ),
    }


def html_audit(frame: pd.DataFrame, html_zip, output, threshold: int) -> dict[str, object]:
    metadata = frame[["sample_id", "serial_number", "label_binary", "split"]].copy()
    metadata["analysis_split"] = np.where(metadata["split"].eq("test"), "test", "train")
    by_serial = {str(int(row.serial_number)): row for row in metadata.itertuples(index=False)}
    rows: list[dict[str, object]] = []
    failed: list[dict[str, str]] = []
    with zipfile.ZipFile(html_zip) as archive:
        files = [item for item in archive.infolist() if not item.is_dir()]
        for number, item in enumerate(files, start=1):
            serial = item.filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            record = by_serial.get(serial)
            if record is None:
                continue
            try:
                content = archive.read(item)
                normalized = normalize_html_bytes(content)
                rows.append({
                    "sample_id": record.sample_id,
                    "serial_number": record.serial_number,
                    "label_binary": record.label_binary,
                    "analysis_split": record.analysis_split,
                    "html_bytes": len(content),
                    "raw_sha256": hashlib.sha256(content).hexdigest(),
                    "normalized_sha256": hashlib.sha256(normalized).hexdigest(),
                    "simhash64": simhash64(normalized),
                })
            except Exception as error:  # preserve the audit trail for corrupt members
                failed.append({"serial_number": serial, "error": f"{type(error).__name__}: {error}"})
            if number % 1000 == 0:
                print(f"Hashed HTML {number}/{len(files)}")
    html = pd.DataFrame(rows)
    pd.DataFrame(failed).to_csv(
        output / "html_read_failures.csv", index=False, encoding="utf-8-sig"
    )

    duplicate_summaries = {}
    for column, filename in (
        ("raw_sha256", "exact_raw_html_cross_split.csv"),
        ("normalized_sha256", "exact_normalized_html_cross_split.csv"),
    ):
        counts = html.groupby(column)["analysis_split"].nunique()
        shared = set(counts[counts > 1].index)
        duplicates = html[html[column].isin(shared)].sort_values(column)
        duplicates.to_csv(output / filename, index=False, encoding="utf-8-sig")
        duplicate_summaries[f"cross_split_{column}_hashes"] = len(shared)
        duplicate_summaries[f"cross_split_{column}_rows"] = len(duplicates)

    train = html[html["analysis_split"].eq("train")]
    test = html[html["analysis_split"].eq("test")]
    buckets: dict[tuple[int, int], list[tuple[int, str, int, int]]] = defaultdict(list)
    for row in train.itertuples(index=False):
        fingerprint = int(row.simhash64)
        for band in range(4):
            buckets[(band, (fingerprint >> (16 * band)) & 0xFFFF)].append(
                (fingerprint, row.sample_id, int(row.serial_number), int(row.label_binary))
            )
    near_rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for row in test.itertuples(index=False):
        fingerprint = int(row.simhash64)
        candidates = set()
        for band in range(4):
            candidates.update(buckets.get((band, (fingerprint >> (16 * band)) & 0xFFFF), []))
        for other_hash, other_id, other_serial, other_label in candidates:
            pair = (other_id, row.sample_id)
            if pair in seen:
                continue
            seen.add(pair)
            distance = (fingerprint ^ other_hash).bit_count()
            if distance <= threshold:
                near_rows.append({
                    "train_sample_id": other_id,
                    "train_serial_number": other_serial,
                    "train_label": other_label,
                    "test_sample_id": row.sample_id,
                    "test_serial_number": int(row.serial_number),
                    "test_label": int(row.label_binary),
                    "hamming_distance": distance,
                    "same_label": other_label == int(row.label_binary),
                })
    near = pd.DataFrame(near_rows)
    if not near.empty:
        near = near.sort_values(["hamming_distance", "train_sample_id", "test_sample_id"])
    near.to_csv(
        output / "near_duplicate_html_cross_split.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return {
        "html_rows_hashed": len(html),
        "html_read_failures": len(failed),
        **duplicate_summaries,
        "near_duplicate_html_pairs": len(near),
        "near_html_hamming_threshold": threshold,
    }


def main() -> None:
    args = arguments()
    output = project_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = read_processed(args.data)
    summary: dict[str, object] = {
        "dataset": "CompPhish v4",
        "train_rows": int(frame[~frame["split"].eq("test")].shape[0]),
        "test_rows": int(frame[frame["split"].eq("test")].shape[0]),
        "model_feature_count": len(ALL_PUBLISHED_FEATURES),
        "feature_protocol": (
            "70 published features; url_more_than_mean and "
            "domain_more_than_mean recomputed from Train phishing rows only"
        ),
        **feature_audit(frame, output, args.nearest_feature_rows),
    }
    if not args.skip_html:
        summary.update(html_audit(
            frame,
            project_path(args.html_zip, must_exist=True),
            output,
            args.near_html_distance,
        ))
    (output / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved leakage audit: {output}")


if __name__ == "__main__":
    main()
