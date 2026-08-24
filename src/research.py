import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(root):
    root = Path(root)

    def run(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    head = run("rev-parse", "HEAD")
    return {
        "available": head is not None,
        "head": head,
        "branch": run("branch", "--show-current") if head else None,
        "dirty": bool(run("status", "--porcelain")) if head else None,
    }


def source_snapshot(root):
    """Hash the reproducibility-relevant source tree even when Git is absent."""
    root = Path(root).resolve()
    candidates = []
    for directory in ("src", "scripts", "tests"):
        candidate = root / directory
        if candidate.exists():
            candidates.extend(path for path in candidate.rglob("*.py") if path.is_file())
    for pattern in ("requirements*.txt", "*.md", "*.toml", "*.yaml", "*.yml"):
        candidates.extend(path for path in root.glob(pattern) if path.is_file())
    files = {}
    for path in sorted(set(candidates)):
        relative = path.relative_to(root).as_posix()
        files[relative] = sha256_file(path)
    digest = hashlib.sha256()
    for relative, file_hash in files.items():
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return {
        "root": str(root),
        "file_count": len(files),
        "sha256": digest.hexdigest(),
        "files": files,
    }


def environment_metadata():
    metadata = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import sklearn

        metadata["scikit_learn"] = sklearn.__version__
    except ImportError:
        metadata["scikit_learn"] = None
    try:
        import torch

        metadata.update({
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
    except ImportError:
        metadata["torch"] = None
    return metadata


def frame_profile(frame):
    profile = {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
    }
    if "label" in frame:
        labels = pd.to_numeric(frame["label"], errors="coerce")
        profile.update({
            "benign": int(labels.eq(0).sum()),
            "phishing": int(labels.eq(1).sum()),
            "invalid_labels": int((~labels.isin([0, 1])).sum()),
        })
    for column in ("html_hash", "template_id", "generator", "generator_family", "brand"):
        if column in frame:
            values = frame[column].fillna("").astype(str)
            profile[f"unique_{column}"] = int(values.nunique())
            profile[f"missing_{column}"] = int(values.str.strip().eq("").sum())
    if "codebase_id" in frame:
        codebases = frame["codebase_id"].fillna("").astype(str)
        profile["codebases"] = int(codebases[codebases.str.strip().ne("")].nunique())
        if "site_label" in frame:
            site_labels = frame.assign(_codebase=codebases).groupby("_codebase")["site_label"]
            profile["site_label_conflicts"] = int(site_labels.nunique(dropna=False).gt(1).sum())
            profile["positive_sites"] = int(site_labels.first().astype(int).eq(1).sum())
    return profile


def evaluation_data_evidence(frame, path):
    """Bind a result file to the exact evaluated split and its provenance."""
    labels = pd.to_numeric(frame["label"], errors="coerce")
    evidence = {
        "path": str(Path(path).resolve()),
        "sha256": sha256_file(path),
        "rows": int(len(frame)),
        "benign": int(labels.eq(0).sum()),
        "phishing": int(labels.eq(1).sum()),
    }
    for column in (
        "dataset_provenance", "dataset_domain", "dataset_license",
        "dataset_citation", "source",
    ):
        if column in frame:
            evidence[column] = sorted(
                frame[column].fillna("unknown").astype(str).str.strip().unique().tolist()
            )
    if "brand" in frame:
        brands = frame["brand"].fillna("").astype(str).str.strip().str.lower()
        known = brands[~brands.isin({"", "unknown", "none", "nan"})]
        evidence["known_brands"] = int(known.nunique())
    if "codebase_id" in frame:
        site_label_column = "site_label" if "site_label" in frame else "label"
        sites = frame[["codebase_id", site_label_column]].copy()
        sites = sites.rename(columns={site_label_column: "_site_label"})
        sites["codebase_id"] = sites["codebase_id"].fillna("").astype(str).str.strip()
        sites = sites[sites["codebase_id"].ne("")]
        conflicts = sites.groupby("codebase_id")["_site_label"].nunique(dropna=False)
        evidence["site_label_conflicts"] = int(conflicts.gt(1).sum())
        site_labels = sites.groupby("codebase_id")["_site_label"].first()
        site_labels = pd.to_numeric(site_labels, errors="coerce")
        evidence["benign_sites"] = int(site_labels.eq(0).sum())
        evidence["phishing_sites"] = int(site_labels.eq(1).sum())
    return evidence


def split_integrity(train, val, test):
    frames = {"train": train, "val": val, "test": test}
    report = {"splits": {name: frame_profile(frame) for name, frame in frames.items()}}
    report["overlap"] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        pair = {}
        for column in ("html_hash", "template_id"):
            if column in frames[left] and column in frames[right]:
                left_values = set(frames[left][column].dropna().astype(str)) - {""}
                right_values = set(frames[right][column].dropna().astype(str)) - {""}
                pair[column] = len(left_values & right_values)
        report["overlap"][f"{left}_{right}"] = pair
    return report


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
