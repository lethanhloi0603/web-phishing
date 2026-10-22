# Copyright (c) 2026 Le Thanh Loi. All rights reserved.
# Author: Le Thanh Loi

"""Shared definitions for leakage-aware CompPhish experiments.

The source workbook contains 70 published variables: five brand-jacking
variables, 42 URL variables, and 23 HTML/content variables.  Two binary URL
variables were computed by the dataset authors using the mean of all phishing
samples.  They are retained for auditability but deliberately excluded from
model inputs because that computation uses label information from the complete
dataset.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


__author__ = "Le Thanh Loi"
__copyright__ = "Copyright (c) 2026 Le Thanh Loi. All rights reserved."
PROJECT_AUTHOR = __author__
PROJECT_REPOSITORY = "https://github.com/lethanhloi0603/web-phishing"


ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42

try:
    import tldextract

    _TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())
except ImportError:
    _TLD_EXTRACTOR = None


BRAND_COLUMNS = [
    "brand_subdomain",
    "brand_substring_domain",
    "brand_typo_dom",
    "fake_tld",
    "brand_in_path_query",
]

URL_COLUMNS_PUBLISHED = [
    "dots_path",
    "url_more_than_mean",
    "domain_more_than_mean",
    "gibberish",
    "domain_length",
    "url_length",
    "url_contains_ip",
    "presence_of_url_shortener",
    "special_characters_count",
    "no_of_at",
    "no_of_comma",
    "no_of_dollar",
    "no_of_semicolon",
    "no_of_space",
    "no_of_ampersand",
    "no_of_dots",
    "no_of_equal",
    "no_of_percent",
    "no_of_underscore",
    "no_of_hyphen",
    "no_of_questionmark",
    "no_of_colon",
    "no_of_slashes_inpath",
    "has_asterisk",
    "has_or",
    "has_embedded_url",
    "uses_https",
    "num_query_params",
    "numeric_proportion",
    "url_entropy",
    "hostname_digit_ratio",
    "has_redirections",
    "has_punycode",
    "count_www",
    "count_com",
    "url_has_high_risk_extension",
    "has_explicit_port",
    "total_words_url",
    "average_length_of_words",
    "longest_word_length",
    "shortest_word_length",
    "presence_of_free_hosting",
]

# These two variables were derived using a label-conditioned global statistic.
LABEL_DERIVED_COLUMNS = ["url_more_than_mean", "domain_more_than_mean"]
URL_COLUMNS = [column for column in URL_COLUMNS_PUBLISHED if column not in LABEL_DERIVED_COLUMNS]

HTML_COLUMNS = [
    "no_of_forms",
    "no_of_Images",
    "no_of_hyperlinks",
    "no_of_external_links",
    "no_of_internal_links",
    "no_of_null_links",
    "no_of_selfreference_links",
    "no_of_empty_links",
    "EmptyToExternalLinksRatio",
    "Risky_extensions_In_Html",
    "presence_of_shorten_url_html",
    "hidden_elements_html",
    "Script_loaded_from_ext_domain",
    "suspicious_inline_js",
    "is_footer_mismatch",
    "is_header_mismatch",
    "is_copyright_mismatch",
    "is_socials_mismatch",
    "is_favicon_mismatch",
    "has_missing_title",
    "title_mismatch_with_domain",
    "has_inline_event_handlers",
    "form_action_external",
]

ALL_PUBLISHED_FEATURES = [*BRAND_COLUMNS, *URL_COLUMNS_PUBLISHED, *HTML_COLUMNS]
ALL_MODEL_FEATURES = [*BRAND_COLUMNS, *URL_COLUMNS, *HTML_COLUMNS]

FEATURE_SETS = {
    "brand": BRAND_COLUMNS,
    "url": URL_COLUMNS,
    "brand_url": [*BRAND_COLUMNS, *URL_COLUMNS],
    "html": HTML_COLUMNS,
    "full": ALL_MODEL_FEATURES,
}

# All 70 published variables.  The two label-conditioned mean indicators must
# not be consumed with their workbook values; FoldSafePhishingMeanFeatures
# below recomputes them from the training portion seen by each Pipeline.fit.
FEATURE_SETS_PUBLISHED = {
    "brand": BRAND_COLUMNS,
    "url": URL_COLUMNS_PUBLISHED,
    "brand_url": [*BRAND_COLUMNS, *URL_COLUMNS_PUBLISHED],
    "html": HTML_COLUMNS,
    "full": ALL_PUBLISHED_FEATURES,
}


class FoldSafePhishingMeanFeatures(BaseEstimator, TransformerMixin):
    """Recompute the two published mean indicators without cross-fold leakage.

    CompPhish defines these flags relative to the mean length of phishing
    samples.  During OOF fitting, ``fit`` therefore learns both thresholds only
    from phishing rows in the current fold's training partition.  ``transform``
    overwrites the published workbook values for train/validation/Test rows.
    """

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series | None = None):
        if y is None:
            raise ValueError(
                "FoldSafePhishingMeanFeatures requires training labels in fit()."
            )
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Fold-safe mean features require pandas DataFrame input.")
        required = {"url_length", "domain_length"}
        missing = sorted(required - set(X.columns))
        if missing:
            raise ValueError(f"Missing source columns for fold-safe features: {missing}")
        labels = np.asarray(y, dtype=int)
        if len(labels) != len(X):
            raise ValueError("X and y have inconsistent row counts.")
        phishing = labels == 1
        if not phishing.any():
            raise ValueError("Training partition contains no phishing samples.")
        self.url_length_threshold_ = float(
            pd.to_numeric(X.loc[phishing, "url_length"], errors="raise").mean()
        )
        self.domain_length_threshold_ = float(
            pd.to_numeric(X.loc[phishing, "domain_length"], errors="raise").mean()
        )
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Fold-safe mean features require pandas DataFrame input.")
        output = X.copy()
        if "url_more_than_mean" in output.columns:
            output["url_more_than_mean"] = (
                pd.to_numeric(output["url_length"], errors="raise")
                > self.url_length_threshold_
            ).astype(np.int8)
        if "domain_more_than_mean" in output.columns:
            output["domain_more_than_mean"] = (
                pd.to_numeric(output["domain_length"], errors="raise")
                > self.domain_length_threshold_
            ).astype(np.int8)
        return output

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return self.feature_names_in_
        return np.asarray(input_features, dtype=object)

# URL cues that a human can often notice directly.  They are valid published
# CompPhish variables, but keeping the list explicit makes shortcut-ablation
# experiments reproducible instead of relying on an undocumented hand-picked
# subset in a notebook.
DIRECTLY_VISIBLE_URL_COLUMNS = [
    "domain_length",
    "url_length",
    "url_contains_ip",
    "presence_of_url_shortener",
    "special_characters_count",
    "no_of_at",
    "no_of_dots",
    "no_of_equal",
    "no_of_percent",
    "no_of_underscore",
    "no_of_hyphen",
    "no_of_questionmark",
    "no_of_colon",
    "no_of_slashes_inpath",
    "uses_https",
    "num_query_params",
    "count_www",
    "count_com",
    "has_explicit_port",
    "total_words_url",
]

REQUIRED_RAW_COLUMNS = {"serial_number", "url", "label", *ALL_PUBLISHED_FEATURES}


def project_path(value: str | Path, *, must_exist: bool = False) -> Path:
    """Resolve a path while preventing writes outside the project."""
    path = Path(value)
    resolved = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ValueError(f"Path must stay inside the project: {resolved}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def normalize_url(value: object) -> str:
    """Conservatively normalize a URL for duplicate detection only."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    candidate = raw if "://" in raw else f"http://{raw}"
    try:
        parts = urlsplit(candidate)
        scheme = parts.scheme.lower() or "http"
        hostname = (parts.hostname or "").strip(".").lower()
        if not hostname:
            return raw.lower()
        try:
            hostname = hostname.encode("idna").decode("ascii")
        except UnicodeError:
            pass
        port = parts.port
        if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
            port = None
        netloc = hostname if port is None else f"{hostname}:{port}"
        path = parts.path or "/"
        return urlunsplit((scheme, netloc, path, parts.query, ""))
    except (TypeError, ValueError):
        return raw.lower()


def hostname_from_url(value: object) -> str:
    raw = str(value or "").strip()
    candidate = raw if "://" in raw else f"http://{raw}"
    try:
        return (urlsplit(candidate).hostname or "").lower().strip(".")
    except ValueError:
        return ""


def registered_domain(value: object) -> str:
    """Return eTLD+1 using tldextract's bundled suffix snapshot."""
    host = str(value or "").strip().lower().strip(".")
    if not host:
        return ""
    try:
        if _TLD_EXTRACTOR is None:
            raise ImportError
        result = _TLD_EXTRACTOR(host)
        return result.top_domain_under_public_suffix or host
    except (ImportError, ValueError):
        labels = [part for part in host.split(".") if part]
        return ".".join(labels[-2:]) if len(labels) >= 2 else host


def make_sample_id(normalized_url: str, serial: object) -> str:
    material = f"{normalized_url}\n{serial}"
    digest = hashlib.sha256(material.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"COMPPHISH_{digest.upper()}"


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    candidates = np.unique(np.quantile(score, np.linspace(0.001, 0.999, 501)))
    return float(max(candidates, key=lambda threshold: f1_score(y, score >= threshold)))


def binary_ece(y: np.ndarray, score: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (score >= lower) & (score < upper if upper < 1.0 else score <= upper)
        if selected.any():
            value += selected.mean() * abs(float(y[selected].mean()) - float(score[selected].mean()))
    return float(value) if len(y) else math.nan


def evaluate_binary(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float | int]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    prediction = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    recall = recall_score(y, prediction, zero_division=0)
    specificity = tn / (tn + fp) if tn + fp else math.nan
    result: dict[str, float | int] = {
        "rows": int(len(y)),
        "accuracy": float((tp + tn) / len(y)) if len(y) else math.nan,
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "balanced_accuracy": float((recall + specificity) / 2),
        "mcc": float(matthews_corrcoef(y, prediction)) if len(np.unique(y)) > 1 else math.nan,
        "fpr": float(fp / (fp + tn)) if fp + tn else math.nan,
        "fnr": float(fn / (fn + tp)) if fn + tp else math.nan,
        "brier": float(np.mean((score - y) ** 2)) if len(y) else math.nan,
        "ece_15": binary_ece(y, score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if len(np.unique(y)) > 1:
        result["pr_auc"] = float(average_precision_score(y, score))
        result["roc_auc"] = float(roc_auc_score(y, score))
    else:
        result["pr_auc"] = math.nan
        result["roc_auc"] = math.nan
    return result


def read_processed(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(project_path(path, must_exist=True), low_memory=False)
    required = {
        "sample_id",
        "url",
        "normalized_url",
        "registered_domain",
        "label_binary",
        "split",
        *ALL_MODEL_FEATURES,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Processed CompPhish file is missing columns: {missing}")
    frame["label_binary"] = pd.to_numeric(frame["label_binary"], errors="raise").astype(int)
    return frame
