import hashlib
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .preprocessing import (
    STATIC_FEATURE_NAMES,
    ARTIFACT_TRANSFORMS,
    apply_artifact_transform,
    dom_tags_to_ids,
    extract_static_features,
    html_to_dom_tags,
    html_to_semantic_text,
    normalize_generator,
    stable_int_id,
)


REQUIRED = {"html", "label", "generator"}


def read_frame(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df.dropna(subset=["html", "label"]).copy()
    df["label"] = df["label"].astype(int)
    if not df["label"].isin([0, 1]).all():
        raise ValueError(f"{path}: label must contain only 0 (benign) and 1 (phishing)")

    df["generator"] = df["generator"].fillna("unknown").astype(str).str.strip().str.lower()
    if "generator_family" not in df:
        df["generator_family"] = df["generator"].map(normalize_generator)
    else:
        df["generator_family"] = (
            df["generator_family"].fillna("unknown").astype(str).map(normalize_generator)
        )
    if "source" not in df:
        df["source"] = "unknown"
    df["source"] = df["source"].fillna("unknown").astype(str).str.lower()
    if "dataset_domain" not in df:
        df["dataset_domain"] = df["source"].map(
            lambda value: "llm_generated" if value == "llm" else "real_world"
        )
    df["dataset_domain"] = df["dataset_domain"].fillna("unknown").astype(str).str.lower()
    for column in ("url", "brand", "template_id"):
        if column not in df:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str)
    return df.reset_index(drop=True)


def build_generator_mapping(df: pd.DataFrame) -> Dict[str, int]:
    families = sorted(df["generator_family"].fillna("unknown").astype(str).unique())
    return {name: index for index, name in enumerate(families)}


class WebPageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        model_name: str = "google/mobilebert-uncased",
        max_html_length: int = 256,
        max_dom_length: int = 256,
        generator_to_id: Optional[Dict[str, int]] = None,
        static_mean=None,
        static_std=None,
        use_html: bool = True,
        use_dom: bool = True,
        use_static: bool = True,
        artifact_augmentation_probability: float = 0.0,
        artifact_augmentation_modes: Sequence[str] = (
            "comments_removed", "identifiers_normalized", "visible_text_removed"
        ),
        augmentation_seed: int = 42,
        return_clean_view: bool = False,
        precomputed_static_features=None,
        precomputed_dom_ids=None,
    ):
        self.df = df.reset_index(drop=True)
        self.max_html_length = max_html_length
        self.max_dom_length = max_dom_length
        self.use_html = use_html
        self.use_dom = use_dom
        self.use_static = use_static
        if not 0.0 <= artifact_augmentation_probability <= 1.0:
            raise ValueError("artifact_augmentation_probability must be between 0 and 1")
        self.artifact_augmentation_probability = float(
            artifact_augmentation_probability
        )
        self.artifact_augmentation_modes = tuple(artifact_augmentation_modes)
        unknown_modes = set(self.artifact_augmentation_modes) - set(ARTIFACT_TRANSFORMS)
        if unknown_modes:
            raise ValueError(
                f"Unknown artifact augmentation modes: {sorted(unknown_modes)}"
            )
        if self.artifact_augmentation_probability and not self.artifact_augmentation_modes:
            raise ValueError("At least one artifact augmentation mode is required")
        self.augmentation_seed = int(augmentation_seed)
        self.augmentation_epoch = 0
        self.return_clean_view = bool(return_clean_view)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) if use_html else None
        self.generator_to_id = generator_to_id or build_generator_mapping(self.df)

        feature_count = len(STATIC_FEATURE_NAMES)
        if use_static and len(self.df):
            if precomputed_static_features is None:
                raw_features = np.vstack([
                    extract_static_features(str(row.html), str(row.url))
                    for row in self.df[["html", "url"]].itertuples(index=False)
                ]).astype(np.float32)
            else:
                raw_features = np.asarray(
                    precomputed_static_features, dtype=np.float32
                )
                if raw_features.shape != (len(self.df), feature_count):
                    raise ValueError(
                        "precomputed_static_features has shape "
                        f"{raw_features.shape}, expected {(len(self.df), feature_count)}"
                    )
            if static_mean is None:
                static_mean = raw_features.mean(axis=0)
            if static_std is None:
                static_std = raw_features.std(axis=0)
            self.static_mean = np.asarray(static_mean, dtype=np.float32)
            self.static_std = np.asarray(static_std, dtype=np.float32)
            self.static_std = np.where(self.static_std < 1e-6, 1.0, self.static_std)
            self.static_features = (raw_features - self.static_mean) / self.static_std
        else:
            self.static_mean = np.zeros(feature_count, dtype=np.float32)
            self.static_std = np.ones(feature_count, dtype=np.float32)
            self.static_features = np.zeros((len(self.df), feature_count), dtype=np.float32)
        self.precomputed_dom_ids = None
        if use_dom and precomputed_dom_ids is not None:
            cached_dom = np.asarray(precomputed_dom_ids, dtype=np.int64)
            if cached_dom.shape != (len(self.df), self.max_dom_length):
                raise ValueError(
                    f"precomputed_dom_ids has shape {cached_dom.shape}, expected "
                    f"{(len(self.df), self.max_dom_length)}"
                )
            self.precomputed_dom_ids = cached_dom

    def __len__(self):
        return len(self.df)

    def set_epoch(self, epoch: int):
        """Select a reproducible, epoch-specific augmentation for every row."""
        self.augmentation_epoch = int(epoch)

    def _artifact_mode(self, idx: int) -> Optional[str]:
        if self.artifact_augmentation_probability <= 0.0:
            return None
        payload = f"{self.augmentation_seed}|{self.augmentation_epoch}|{idx}".encode()
        digest = hashlib.sha256(payload).digest()
        draw = int.from_bytes(digest[:8], "big") / float(2**64)
        if draw >= self.artifact_augmentation_probability:
            return None
        mode_index = int.from_bytes(digest[8:16], "big") % len(
            self.artifact_augmentation_modes
        )
        return self.artifact_augmentation_modes[mode_index]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clean_html = str(row["html"])
        artifact_mode = self._artifact_mode(idx)
        raw_html = clean_html
        if artifact_mode is not None:
            raw_html = apply_artifact_transform(raw_html, artifact_mode)

        def encode_view(view_html, clean=False):
            values = {}
            if self.use_html:
                tokenized = self.tokenizer(
                    html_to_semantic_text(view_html),
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_html_length,
                    return_tensors="pt",
                )
                values["input_ids"] = tokenized["input_ids"].squeeze(0)
                values["attention_mask"] = tokenized["attention_mask"].squeeze(0)
            else:
                values["input_ids"] = torch.zeros(1, dtype=torch.long)
                values["attention_mask"] = torch.ones(1, dtype=torch.long)

            if self.use_dom:
                if clean and self.precomputed_dom_ids is not None:
                    values["dom_ids"] = torch.tensor(
                        self.precomputed_dom_ids[idx], dtype=torch.long
                    )
                else:
                    values["dom_ids"] = torch.tensor(
                        dom_tags_to_ids(
                            html_to_dom_tags(view_html, self.max_dom_length),
                            self.max_dom_length,
                        ),
                        dtype=torch.long,
                    )
                values["dom_mask"] = (values["dom_ids"] != 0).long()
            else:
                values["dom_ids"] = torch.zeros(1, dtype=torch.long)
                values["dom_mask"] = torch.ones(1, dtype=torch.long)

            if self.use_static:
                static = (
                    self.static_features[idx]
                    if clean
                    else (
                        extract_static_features(view_html, str(row["url"]))
                        - self.static_mean
                    )
                    / self.static_std
                )
            else:
                static = self.static_features[idx]
            values["static_features"] = torch.tensor(static, dtype=torch.float32)
            return values

        view = encode_view(raw_html, clean=artifact_mode is None)

        family = str(row["generator_family"])
        template = str(row.get("template_id", "")) or "unknown-template"
        brand = str(row.get("brand", "")) or "unknown-brand"
        dataset_domain = str(row.get("dataset_domain", "unknown")).lower()
        source = str(row.get("source", "unknown")).lower()
        domain_id = 1 if dataset_domain == "llm_generated" or source == "llm" else 0

        result = {
            **view,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "generator_id": torch.tensor(self.generator_to_id.get(family, -1), dtype=torch.long),
            "domain_id": torch.tensor(domain_id, dtype=torch.long),
            "template_id": torch.tensor(stable_int_id(template), dtype=torch.long),
            "brand_id": torch.tensor(stable_int_id(brand), dtype=torch.long),
            "generator_name": family,
            "artifact_augmentation": artifact_mode or "original",
            "artifact_applied": torch.tensor(
                float(artifact_mode is not None), dtype=torch.float32
            ),
        }
        if self.return_clean_view:
            clean_view = encode_view(clean_html, clean=True)
            result.update({f"clean_{key}": value for key, value in clean_view.items()})
        return result


def precompute_page_features(df, max_dom_length, use_dom=True, use_static=True):
    """Compute reusable raw DOM/static features for repeated checkpoint evaluation."""
    static = None
    dom = None
    if use_static:
        static = np.vstack([
            extract_static_features(str(row.html), str(row.url))
            for row in df[["html", "url"]].itertuples(index=False)
        ]).astype(np.float32)
    if use_dom:
        dom = np.asarray([
            dom_tags_to_ids(
                html_to_dom_tags(str(html), max_dom_length), max_dom_length
            )
            for html in df["html"].astype(str)
        ], dtype=np.int64)
    return static, dom
