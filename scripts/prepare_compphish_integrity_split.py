# Copyright (c) 2026 Le Thanh Loi. All rights reserved.
# Author: Le Thanh Loi

"""Create an integrity-clean, component-disjoint CompPhish Train/Test split.

Samples are connected when they share at least one of the following:

1. registered domain;
2. the complete 68-feature vector used for modelling;
3. whitespace-normalized raw HTML.

Connected components, rather than individual rows, are assigned to Train/Test.
This prevents transitive leakage: if A shares a domain with B and B shares an
HTML template with C, all three remain in the same set. The existing processed
file is read but never overwritten by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from compphish_common import ALL_MODEL_FEATURES, project_path, read_processed


WHITESPACE_PATTERN = re.compile(rb"\s+")


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, value: int) -> int:
        parent = int(self.parent[value])
        if parent != value:
            self.parent[value] = self.find(parent)
        return int(self.parent[value])

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/processed/compphish/compphish_clean_split.csv.gz",
        help="Previously cleaned CompPhish table; its old split is ignored.",
    )
    parser.add_argument("--html-zip", default="data/raw/CompPhish/All_HTML.zip")
    parser.add_argument(
        "--output",
        default="data/processed/compphish/compphish_integrity_split.csv.gz",
    )
    parser.add_argument(
        "--audit-dir", default="artifacts/compphish/integrity_split_audit"
    )
    parser.add_argument("--test-folds", type=int, default=3)
    parser.add_argument("--low-information-bytes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_html(content: bytes) -> bytes:
    return WHITESPACE_PATTERN.sub(b" ", content.strip().lower())


def feature_hashes(frame: pd.DataFrame) -> list[str]:
    values = frame[ALL_MODEL_FEATURES].apply(pd.to_numeric, errors="raise")
    matrix = np.ascontiguousarray(values.to_numpy(dtype="<f8"))
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in matrix]


def html_metadata(frame: pd.DataFrame, archive_path) -> pd.DataFrame:
    serial_to_index = {
        str(int(serial)): index for index, serial in enumerate(frame["serial_number"])
    }
    records: list[dict[str, object]] = []
    seen_indices: set[int] = set()
    with zipfile.ZipFile(archive_path) as archive:
        files = [item for item in archive.infolist() if not item.is_dir()]
        for number, item in enumerate(files, start=1):
            serial = item.filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            index = serial_to_index.get(serial)
            if index is None:
                continue
            content = archive.read(item)
            normalized = normalize_html(content)
            records.append({
                "row_index": index,
                "serial_number": frame.iloc[index]["serial_number"],
                "html_bytes": len(content),
                "normalized_html_bytes": len(normalized),
                "raw_html_sha256": hashlib.sha256(content).hexdigest(),
                "normalized_html_sha256": hashlib.sha256(normalized).hexdigest(),
            })
            seen_indices.add(index)
            if number % 1000 == 0:
                print(f"Hashed HTML {number}/{len(files)}")
    for index in sorted(set(range(len(frame))) - seen_indices):
        serial = frame.iloc[index]["serial_number"]
        records.append({
            "row_index": index,
            "serial_number": serial,
            "html_bytes": np.nan,
            "normalized_html_bytes": np.nan,
            "raw_html_sha256": f"missing-raw-{serial}",
            "normalized_html_sha256": f"missing-normalized-{serial}",
        })
    return pd.DataFrame(records).sort_values("row_index").reset_index(drop=True)


def union_equal_keys(disjoint: DisjointSet, keys: list[str] | pd.Series) -> int:
    first_position: dict[str, int] = {}
    unions = 0
    for position, key_value in enumerate(keys):
        key = str(key_value)
        first = first_position.get(key)
        if first is None:
            first_position[key] = position
        else:
            before = disjoint.find(position) != disjoint.find(first)
            disjoint.union(position, first)
            unions += int(before)
    return unions


def stable_component_ids(frame: pd.DataFrame, disjoint: DisjointSet) -> pd.Series:
    members: dict[int, list[str]] = {}
    for position, sample_id in enumerate(frame["sample_id"].astype(str)):
        members.setdefault(disjoint.find(position), []).append(sample_id)
    identifiers = {
        root: "COMP_" + hashlib.sha256(
            "\n".join(sorted(sample_ids)).encode("utf-8")
        ).hexdigest()[:16].upper()
        for root, sample_ids in members.items()
    }
    return pd.Series(
        [identifiers[disjoint.find(position)] for position in range(len(frame))],
        index=frame.index,
        name="integrity_component",
    )


def assign_splits(
    frame: pd.DataFrame, component: pd.Series, test_folds: int, seed: int
) -> np.ndarray:
    if not 1 <= test_folds <= 10:
        raise ValueError("--test-folds must be between 1 and 10")
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    fold_id = np.full(len(frame), -1, dtype=int)
    y = frame["label_binary"].to_numpy(dtype=int)
    for fold_number, (_, holdout) in enumerate(
        splitter.split(frame, y, groups=component.to_numpy())
    ):
        fold_id[holdout] = fold_number
    if (fold_id < 0).any():
        raise RuntimeError("At least one row was not assigned to a fold")
    return np.where(fold_id < 20 - test_folds, "train", "test")


def cross_split_overlap(frame: pd.DataFrame, column: str) -> int:
    grouped = frame.groupby(column)["split"].nunique()
    return int((grouped > 1).sum())


def main() -> None:
    args = arguments()
    source = project_path(args.input, must_exist=True)
    html_zip = project_path(args.html_zip, must_exist=True)
    destination = project_path(args.output)
    audit_dir = project_path(args.audit_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    frame = read_processed(source).reset_index(drop=True)
    # The old assignment is retained for auditability but never used to create
    # the new split.
    frame = frame.rename(columns={"split": "original_split"})
    frame["feature_vector_sha256"] = feature_hashes(frame)
    html = html_metadata(frame, html_zip)
    if not np.array_equal(html["row_index"].to_numpy(), np.arange(len(frame))):
        raise RuntimeError("HTML metadata did not align with processed rows")
    for column in (
        "html_bytes", "normalized_html_bytes", "raw_html_sha256",
        "normalized_html_sha256",
    ):
        frame[column] = html[column].to_numpy()
    frame["html_low_information"] = (
        frame["normalized_html_bytes"].fillna(0) < args.low_information_bytes
    ).astype(int)

    disjoint = DisjointSet(len(frame))
    domain_unions = union_equal_keys(disjoint, frame["registered_domain"])
    feature_unions = union_equal_keys(disjoint, frame["feature_vector_sha256"])
    html_unions = union_equal_keys(disjoint, frame["normalized_html_sha256"])
    frame["integrity_component"] = stable_component_ids(frame, disjoint)
    frame["split"] = assign_splits(
        frame, frame["integrity_component"], args.test_folds, args.seed
    )

    overlap = {
        "registered_domain": cross_split_overlap(frame, "registered_domain"),
        "feature_vector_sha256": cross_split_overlap(frame, "feature_vector_sha256"),
        "raw_html_sha256": cross_split_overlap(frame, "raw_html_sha256"),
        "normalized_html_sha256": cross_split_overlap(frame, "normalized_html_sha256"),
        "integrity_component": cross_split_overlap(frame, "integrity_component"),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Integrity split still contains cross-set overlap: {overlap}")

    component_table = frame.groupby("integrity_component", as_index=False).agg(
        rows=("sample_id", "size"),
        phishing=("label_binary", "sum"),
        domains=("registered_domain", "nunique"),
        feature_vectors=("feature_vector_sha256", "nunique"),
        normalized_html=("normalized_html_sha256", "nunique"),
        low_information_html=("html_low_information", "sum"),
        split=("split", "first"),
    )
    component_table["legitimate"] = component_table["rows"] - component_table["phishing"]
    component_table.to_csv(
        audit_dir / "integrity_components.csv", index=False, encoding="utf-8-sig"
    )

    split_label = pd.crosstab(frame["split"], frame["label_binary"], margins=True)
    split_label = split_label.rename(columns={0: "legitimate", 1: "phishing"})
    split_label.to_csv(audit_dir / "split_label.csv", encoding="utf-8-sig")
    frame.loc[frame["html_low_information"].eq(1), [
        "sample_id", "serial_number", "registered_domain", "label_binary",
        "split", "html_bytes", "normalized_html_bytes", "raw_html_sha256",
    ]].to_csv(
        audit_dir / "low_information_html.csv", index=False, encoding="utf-8-sig"
    )

    # Preserve all published/model columns plus integrity metadata. Downstream
    # readers ignore additional audit columns and consume the new split.
    frame.to_csv(
        destination, index=False, encoding="utf-8-sig", compression="gzip"
    )

    component_sizes = component_table["rows"]
    mixed_components = int(
        ((component_table["phishing"] > 0) & (component_table["legitimate"] > 0)).sum()
    )
    summary = {
        "dataset": "CompPhish v4",
        "rows": len(frame),
        "model_feature_count": len(ALL_MODEL_FEATURES),
        "split_counts": frame["split"].value_counts().to_dict(),
        "split_label_counts": {
            split: {
                "legitimate": int(subset["label_binary"].eq(0).sum()),
                "phishing": int(subset["label_binary"].eq(1).sum()),
            }
            for split, subset in frame.groupby("split")
        },
        "integrity_components": int(frame["integrity_component"].nunique()),
        "largest_component_rows": int(component_sizes.max()),
        "median_component_rows": float(component_sizes.median()),
        "mixed_label_components": mixed_components,
        "low_information_html_rows": int(frame["html_low_information"].sum()),
        "union_operations": {
            "registered_domain": domain_unions,
            "exact_feature_vector": feature_unions,
            "exact_normalized_html": html_unions,
        },
        "cross_split_overlap_after_grouping": overlap,
        "split_method": (
            "connected components of registered domain + exact 68-feature vector + "
            "exact normalized HTML; 20-fold StratifiedGroupKFold with "
            f"{20 - args.test_folds}/{args.test_folds} folds for Train/Test"
        ),
        "old_split_used": False,
        "seed": args.seed,
    }
    (audit_dir / "integrity_split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved integrity-clean data: {destination}")
    print(f"Saved split audit: {audit_dir}")


if __name__ == "__main__":
    main()
