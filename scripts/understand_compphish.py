# Copyright (c) 2026 Le Thanh Loi. All rights reserved.
# Author: Le Thanh Loi

"""Create a thesis-ready Data Understanding package for CompPhish.

The script is deliberately analysis-only: it does not train or select a final
phishing detector.  Structural quality checks use the complete cleaned corpus,
while every feature-to-label association is computed on the training portion
only.  Existing ``train`` and ``val`` rows are merged into one analysis-train
set so the report follows a two-set Train/Test presentation without changing
the fixed domain-disjoint test set.

Outputs include CSV tables, a JSON summary, a Markdown report, and PNG figures.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score, roc_curve

from compphish_common import (
    ALL_PUBLISHED_FEATURES,
    BRAND_COLUMNS,
    HTML_COLUMNS,
    LABEL_DERIVED_COLUMNS,
    URL_COLUMNS_PUBLISHED,
    project_path,
    read_processed,
)


LABEL_NAMES = {0: "Legitimate", 1: "Phishing"}
COLORS = {0: "#2F6690", 1: "#C73E1D"}

# These are surface cues that a person can usually inspect directly in the URL
# without loading the page or comparing it with webpage content.  The taxonomy
# is an explicit analysis convention, not a claim that the features are invalid.
DIRECTLY_VISIBLE_URL_FEATURES = [
    "domain_length",
    "url_length",
    "url_contains_ip",
    "presence_of_url_shortener",
    "special_characters_count",
    "no_of_at",
    "no_of_dots",
    "no_of_hyphen",
    "no_of_questionmark",
    "no_of_slashes_inpath",
    "uses_https",
    "has_punycode",
    "count_www",
    "count_com",
    "url_has_high_risk_extension",
    "has_explicit_port",
    "presence_of_free_hosting",
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/processed/compphish/compphish_clean_split.csv.gz",
        help="Clean CompPhish CSV created by prepare_compphish.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/compphish/data_understanding",
    )
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def feature_group(feature: str) -> str:
    if feature in BRAND_COLUMNS:
        return "Brand-jacking"
    if feature in URL_COLUMNS_PUBLISHED:
        return "URL"
    if feature in HTML_COLUMNS:
        return "HTML"
    return "Other"


def visibility_tier(feature: str) -> str:
    if feature in DIRECTLY_VISIBLE_URL_FEATURES:
        return "directly_visible_url"
    if feature in LABEL_DERIVED_COLUMNS:
        return "label_conditioned_threshold"
    if feature in HTML_COLUMNS:
        return "html_or_relational"
    if feature in BRAND_COLUMNS:
        return "brand_comparison"
    return "computed_url"


def safe_float(value: object) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame as Markdown without requiring tabulate."""

    def format_cell(value: object) -> str:
        if pd.isna(value):
            text = ""
        elif isinstance(value, (float, np.floating)):
            text = f"{float(value):.6f}".rstrip("0").rstrip(".")
        else:
            text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    headers = [format_cell(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(format_cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *rows,
        ]
    )


def save_figure(fig: plt.Figure, destination: Path, dpi: int) -> None:
    fig.savefig(destination, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def two_set_view(frame: pd.DataFrame) -> pd.Series:
    original = frame["split"].astype(str).str.lower()
    unknown = sorted(set(original) - {"train", "val", "validation", "test"})
    if unknown:
        raise ValueError(f"Unsupported split labels: {unknown}")
    return pd.Series(
        np.where(original.eq("test"), "test", "train"),
        index=frame.index,
        name="analysis_split",
    )


def structural_tables(frame: pd.DataFrame, output: Path) -> dict[str, object]:
    label_table = (
        frame.assign(label_name=frame["label_binary"].map(LABEL_NAMES))
        .groupby("label_name", observed=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    label_table["percentage"] = 100.0 * label_table["rows"] / len(frame)
    label_table.to_csv(
        output / "table_01_label_distribution.csv", index=False, encoding="utf-8-sig"
    )

    split_table = pd.crosstab(
        frame["analysis_split"], frame["label_binary"], margins=False
    ).rename(columns=LABEL_NAMES)
    for label in LABEL_NAMES.values():
        if label not in split_table:
            split_table[label] = 0
    split_table = split_table[["Legitimate", "Phishing"]]
    split_table["Total"] = split_table.sum(axis=1)
    split_table["Percentage"] = 100.0 * split_table["Total"] / len(frame)
    split_table.to_csv(
        output / "table_02_train_test_distribution.csv", encoding="utf-8-sig"
    )

    group_table = pd.DataFrame(
        [
            {"feature_group": "Brand-jacking", "feature_count": len(BRAND_COLUMNS)},
            {"feature_group": "URL", "feature_count": len(URL_COLUMNS_PUBLISHED)},
            {"feature_group": "HTML", "feature_count": len(HTML_COLUMNS)},
        ]
    )
    group_table.to_csv(
        output / "table_03_feature_groups.csv", index=False, encoding="utf-8-sig"
    )

    quality_rows = []
    for feature in ALL_PUBLISHED_FEATURES:
        values = pd.to_numeric(frame[feature], errors="coerce")
        quality_rows.append(
            {
                "feature": feature,
                "feature_group": feature_group(feature),
                "visibility_tier": visibility_tier(feature),
                "missing_count": int(values.isna().sum()),
                "missing_percentage": 100.0 * float(values.isna().mean()),
                "unique_values": int(values.nunique(dropna=True)),
                "minimum": safe_float(values.min()) if values.notna().any() else None,
                "maximum": safe_float(values.max()) if values.notna().any() else None,
                "mean": safe_float(values.mean()) if values.notna().any() else None,
                "median": safe_float(values.median()) if values.notna().any() else None,
            }
        )
    quality = pd.DataFrame(quality_rows)
    quality.to_csv(
        output / "table_04_feature_quality.csv", index=False, encoding="utf-8-sig"
    )

    train_domains = set(
        frame.loc[frame["analysis_split"].eq("train"), "registered_domain"].astype(str)
    )
    test_domains = set(
        frame.loc[frame["analysis_split"].eq("test"), "registered_domain"].astype(str)
    )
    duplicate_urls = int(frame["normalized_url"].duplicated().sum())
    duplicate_domains = int(frame["registered_domain"].duplicated().sum())

    return {
        "rows": int(len(frame)),
        "labels": {
            "legitimate": int(frame["label_binary"].eq(0).sum()),
            "phishing": int(frame["label_binary"].eq(1).sum()),
        },
        "analysis_splits": {
            name: int(frame["analysis_split"].eq(name).sum())
            for name in ("train", "test")
        },
        "published_features": int(len(ALL_PUBLISHED_FEATURES)),
        "feature_groups": {
            "brand_jacking": int(len(BRAND_COLUMNS)),
            "url": int(len(URL_COLUMNS_PUBLISHED)),
            "html": int(len(HTML_COLUMNS)),
        },
        "unique_registered_domains": int(frame["registered_domain"].nunique()),
        "train_test_registered_domain_overlap": int(len(train_domains & test_domains)),
        "duplicate_normalized_urls_in_clean_file": duplicate_urls,
        "repeated_registered_domain_rows": duplicate_domains,
        "missing_feature_values": int(
            frame[ALL_PUBLISHED_FEATURES].isna().sum().sum()
        ),
    }


def label_summary_and_association(
    train: pd.DataFrame, output: Path, seed: int
) -> pd.DataFrame:
    summary_rows = []
    association_rows = []
    y = train["label_binary"].to_numpy(dtype=int)

    for feature in ALL_PUBLISHED_FEATURES:
        raw = pd.to_numeric(train[feature], errors="coerce")
        median = float(raw.median()) if raw.notna().any() else 0.0
        values = raw.fillna(median).to_numpy(dtype=float)
        legitimate = values[y == 0]
        phishing = values[y == 1]

        for label, subset in ((0, legitimate), (1, phishing)):
            summary_rows.append(
                {
                    "feature": feature,
                    "feature_group": feature_group(feature),
                    "visibility_tier": visibility_tier(feature),
                    "label": LABEL_NAMES[label],
                    "rows": int(len(subset)),
                    "mean": safe_float(np.mean(subset)),
                    "std": safe_float(np.std(subset, ddof=1)) if len(subset) > 1 else None,
                    "median": safe_float(np.median(subset)),
                    "q25": safe_float(np.quantile(subset, 0.25)),
                    "q75": safe_float(np.quantile(subset, 0.75)),
                    "positive_rate": safe_float(np.mean(subset > 0)),
                }
            )

        unique_values = int(np.unique(values).size)
        if unique_values < 2:
            raw_auc = math.nan
            direction_free_auc = math.nan
            best_balanced_accuracy = math.nan
            direction = "constant"
            mutual_information = 0.0
        else:
            raw_auc = float(roc_auc_score(y, values))
            direction_free_auc = max(raw_auc, 1.0 - raw_auc)
            oriented = values if raw_auc >= 0.5 else -values
            fpr, tpr, _ = roc_curve(y, oriented)
            best_balanced_accuracy = float(np.max((tpr + 1.0 - fpr) / 2.0))
            direction = "higher_phishing" if raw_auc >= 0.5 else "lower_phishing"
            is_discrete = unique_values <= 20
            mutual_information = float(
                mutual_info_classif(
                    values.reshape(-1, 1),
                    y,
                    discrete_features=[is_discrete],
                    random_state=seed,
                )[0]
            )

        pooled_variance = (
            ((len(legitimate) - 1) * np.var(legitimate, ddof=1))
            + ((len(phishing) - 1) * np.var(phishing, ddof=1))
        ) / max(len(legitimate) + len(phishing) - 2, 1)
        pooled_std = math.sqrt(max(float(pooled_variance), 0.0))
        standardized_difference = (
            float(np.mean(phishing) - np.mean(legitimate)) / pooled_std
            if pooled_std > 0
            else 0.0
        )

        association_rows.append(
            {
                "feature": feature,
                "feature_group": feature_group(feature),
                "visibility_tier": visibility_tier(feature),
                "unique_values_train": unique_values,
                "legitimate_mean": safe_float(np.mean(legitimate)),
                "phishing_mean": safe_float(np.mean(phishing)),
                "mean_difference_phishing_minus_legitimate": safe_float(
                    np.mean(phishing) - np.mean(legitimate)
                ),
                "standardized_difference": safe_float(standardized_difference),
                "raw_roc_auc": safe_float(raw_auc),
                "direction_free_roc_auc": safe_float(direction_free_auc),
                "best_univariate_balanced_accuracy_train": safe_float(
                    best_balanced_accuracy
                ),
                "mutual_information": safe_float(mutual_information),
                "direction": direction,
            }
        )

    pd.DataFrame(summary_rows).to_csv(
        output / "table_05_feature_summary_by_label_train.csv",
        index=False,
        encoding="utf-8-sig",
    )
    associations = pd.DataFrame(association_rows).sort_values(
        ["direction_free_roc_auc", "mutual_information"],
        ascending=False,
        na_position="last",
    )
    associations.to_csv(
        output / "table_06_univariate_association_train.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # A candidate is a cue worth auditing, not a feature automatically removed.
    shortcut_candidates = associations[
        associations["direction_free_roc_auc"].fillna(0.5).ge(0.70)
        | associations["best_univariate_balanced_accuracy_train"].fillna(0.5).ge(0.70)
    ].copy()
    shortcut_candidates["audit_reason"] = np.where(
        shortcut_candidates["visibility_tier"].eq("directly_visible_url"),
        "strong_univariate_and_directly_visible",
        "strong_univariate_association",
    )
    shortcut_candidates.to_csv(
        output / "table_07_shortcut_candidates_train.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return associations


def label_conditioned_feature_audit(
    frame: pd.DataFrame, output: Path
) -> tuple[pd.DataFrame, dict[str, float]]:
    train = frame[frame["analysis_split"].eq("train")]
    phishing_train = train[train["label_binary"].eq(1)]
    thresholds = {
        "url_more_than_mean": float(phishing_train["url_length"].mean()),
        "domain_more_than_mean": float(phishing_train["domain_length"].mean()),
    }
    source_columns = {
        "url_more_than_mean": "url_length",
        "domain_more_than_mean": "domain_length",
    }
    rows = []
    for feature, source in source_columns.items():
        recomputed = (
            pd.to_numeric(frame[source], errors="coerce").fillna(0) > thresholds[feature]
        ).astype(int)
        published = pd.to_numeric(frame[feature], errors="coerce").fillna(0).astype(int)
        for split in ("train", "test"):
            selected = frame["analysis_split"].eq(split)
            for label in (0, 1):
                mask = selected & frame["label_binary"].eq(label)
                rows.append(
                    {
                        "feature": feature,
                        "source_measure": source,
                        "train_phishing_threshold": thresholds[feature],
                        "analysis_split": split,
                        "label": LABEL_NAMES[label],
                        "rows": int(mask.sum()),
                        "published_positive_rate": float(published[mask].mean()),
                        "train_rule_positive_rate": float(recomputed[mask].mean()),
                        "published_vs_train_rule_agreement": float(
                            (published[mask] == recomputed[mask]).mean()
                        ),
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(
        output / "table_08_label_conditioned_feature_check.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return table, thresholds


def count_www_audit(frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    train = frame[frame["analysis_split"].eq("train")].copy()
    train["count_www"] = pd.to_numeric(train["count_www"], errors="coerce").fillna(0)
    train["has_www_feature"] = train["count_www"].gt(0)
    train["raw_url_contains_www"] = train["url"].astype(str).str.contains(
        "www", case=False, regex=False
    )
    rows = []
    for label in (0, 1):
        subset = train[train["label_binary"].eq(label)]
        rows.append(
            {
                "label": LABEL_NAMES[label],
                "rows": int(len(subset)),
                "count_www_mean": float(subset["count_www"].mean()),
                "count_www_median": float(subset["count_www"].median()),
                "count_www_positive_rate": float(subset["has_www_feature"].mean()),
                "raw_url_contains_www_rate": float(
                    subset["raw_url_contains_www"].mean()
                ),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(
        output / "table_09_count_www_audit_train.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return table


def plot_label_distribution(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    counts = frame["label_binary"].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(
        [LABEL_NAMES[index] for index in counts.index],
        counts.values,
        color=[COLORS[index] for index in counts.index],
        width=0.58,
    )
    ax.bar_label(bars, labels=[f"{value:,}" for value in counts.values], padding=4)
    ax.set_ylabel("Number of samples")
    ax.set_title("CompPhish class distribution after cleaning")
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, output / "figure_01_class_distribution.png", dpi)


def plot_train_test(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    table = pd.crosstab(frame["analysis_split"], frame["label_binary"]).reindex(
        index=["train", "test"], columns=[0, 1], fill_value=0
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    x = np.arange(len(table.index))
    width = 0.34
    first = ax.bar(x - width / 2, table[0], width, label="Legitimate", color=COLORS[0])
    second = ax.bar(x + width / 2, table[1], width, label="Phishing", color=COLORS[1])
    ax.bar_label(first, padding=3, fmt="%d")
    ax.bar_label(second, padding=3, fmt="%d")
    ax.set_xticks(x, ["Train", "Test"])
    ax.set_ylabel("Number of samples")
    ax.set_title("Domain-disjoint Train/Test distribution")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, output / "figure_02_train_test_distribution.png", dpi)


def plot_feature_groups(output: Path, dpi: int) -> None:
    labels = ["Brand-jacking", "URL", "HTML"]
    counts = [len(BRAND_COLUMNS), len(URL_COLUMNS_PUBLISHED), len(HTML_COLUMNS)]
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    bars = ax.bar(labels, counts, color=["#7A5195", "#2F6690", "#4C956C"], width=0.58)
    ax.bar_label(bars, padding=4)
    ax.set_ylabel("Number of published features")
    ax.set_title("CompPhish feature groups (70 features)")
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, output / "figure_03_feature_groups.png", dpi)


def plot_count_www(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    train = frame[frame["analysis_split"].eq("train")].copy()
    train["count_www"] = pd.to_numeric(train["count_www"], errors="coerce").fillna(0)
    rates = train.groupby("label_binary")["count_www"].apply(lambda x: (x > 0).mean())
    rates = rates.reindex([0, 1], fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    bars = axes[0].bar(
        [LABEL_NAMES[index] for index in rates.index],
        100 * rates.values,
        color=[COLORS[index] for index in rates.index],
        width=0.58,
    )
    axes[0].bar_label(bars, labels=[f"{100 * value:.1f}%" for value in rates], padding=4)
    axes[0].set_ylim(0, max(100.0, float(100 * rates.max() * 1.15)))
    axes[0].set_ylabel("URLs with count_www > 0 (%)")
    axes[0].set_title("Presence of www by class (Train only)")
    axes[0].spines[["top", "right"]].set_visible(False)

    box_data = [
        train.loc[train["label_binary"].eq(label), "count_www"].to_numpy()
        for label in (0, 1)
    ]
    boxes = axes[1].boxplot(
        box_data,
        tick_labels=["Legitimate", "Phishing"],
        patch_artist=True,
        showfliers=False,
    )
    for patch, color in zip(boxes["boxes"], [COLORS[0], COLORS[1]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[1].set_ylabel("count_www")
    axes[1].set_title("count_www distribution (Train only)")
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, output / "figure_04_count_www_audit.png", dpi)


def plot_top_associations(
    associations: pd.DataFrame, output: Path, top_k: int, dpi: int
) -> None:
    table = associations.dropna(subset=["direction_free_roc_auc"]).head(top_k).copy()
    table = table.sort_values("direction_free_roc_auc")
    colors = [
        "#C73E1D" if tier == "directly_visible_url" else "#2F6690"
        for tier in table["visibility_tier"]
    ]
    fig, ax = plt.subplots(figsize=(9.0, max(5.0, 0.38 * len(table))))
    ax.barh(table["feature"], table["direction_free_roc_auc"], color=colors)
    ax.axvline(0.5, color="#555555", linestyle="--", linewidth=1)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Direction-free univariate ROC-AUC (Train only)")
    ax.set_title("Strongest single-feature associations")
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, output / "figure_05_top_univariate_auc.png", dpi)


def plot_length_distributions(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    train = frame[frame["analysis_split"].eq("train")]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for axis, feature, title in (
        (axes[0], "url_length", "URL length"),
        (axes[1], "domain_length", "Domain length"),
    ):
        values_all = pd.to_numeric(train[feature], errors="coerce").dropna()
        upper = float(values_all.quantile(0.99)) if len(values_all) else 1.0
        bins = np.linspace(0, max(upper, 1.0), 35)
        for label in (0, 1):
            values = pd.to_numeric(
                train.loc[train["label_binary"].eq(label), feature], errors="coerce"
            ).dropna()
            axis.hist(
                values.clip(upper=upper),
                bins=bins,
                density=True,
                alpha=0.45,
                color=COLORS[label],
                label=LABEL_NAMES[label],
            )
        axis.set_xlabel(feature)
        axis.set_ylabel("Density")
        axis.set_title(f"{title} by class (capped at 99th percentile)")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output / "figure_06_length_distributions.png", dpi)


def plot_correlation_heatmap(
    frame: pd.DataFrame,
    associations: pd.DataFrame,
    output: Path,
    dpi: int,
    size: int = 12,
) -> None:
    train = frame[frame["analysis_split"].eq("train")]
    features = (
        associations.dropna(subset=["direction_free_roc_auc"])
        .head(size)["feature"]
        .tolist()
    )
    values = train[features].apply(pd.to_numeric, errors="coerce")
    values = values.fillna(values.median())
    correlation = values.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(10.0, 8.5))
    image = ax.imshow(correlation.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(features)), features, rotation=55, ha="right")
    ax.set_yticks(np.arange(len(features)), features)
    ax.set_title("Spearman correlation among top univariate features (Train only)")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Spearman correlation")
    fig.tight_layout()
    save_figure(fig, output / "figure_07_top_feature_correlation.png", dpi)


def plot_shortcut_map(
    associations: pd.DataFrame, output: Path, dpi: int
) -> None:
    table = associations.dropna(
        subset=["direction_free_roc_auc", "mutual_information"]
    ).copy()
    direct = table["visibility_tier"].eq("directly_visible_url")
    fig, ax = plt.subplots(figsize=(8.0, 5.3))
    ax.scatter(
        table.loc[~direct, "direction_free_roc_auc"],
        table.loc[~direct, "mutual_information"],
        color="#2F6690",
        alpha=0.7,
        label="Other features",
    )
    ax.scatter(
        table.loc[direct, "direction_free_roc_auc"],
        table.loc[direct, "mutual_information"],
        color="#C73E1D",
        alpha=0.85,
        label="Directly visible URL cues",
    )
    for _, row in table.nlargest(8, "direction_free_roc_auc").iterrows():
        ax.annotate(
            row["feature"],
            (row["direction_free_roc_auc"], row["mutual_information"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axvline(0.70, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("Direction-free univariate ROC-AUC (Train only)")
    ax.set_ylabel("Mutual information")
    ax.set_title("Feature-association map for shortcut auditing")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, output / "figure_08_shortcut_audit_map.png", dpi)


def write_report(
    output: Path,
    structural: dict[str, object],
    associations: pd.DataFrame,
    count_www: pd.DataFrame,
    thresholds: dict[str, float],
) -> None:
    top = associations.head(10)[
        [
            "feature",
            "feature_group",
            "visibility_tier",
            "direction_free_roc_auc",
            "mutual_information",
            "direction",
        ]
    ]
    report = [
        "# CompPhish Data Understanding Report",
        "",
        "Feature-to-label analyses in this package use Train only. The fixed Test set is not used for feature screening.",
        "",
        "## Dataset overview",
        "",
        f"- Clean rows: {structural['rows']:,}",
        f"- Legitimate: {structural['labels']['legitimate']:,}",
        f"- Phishing: {structural['labels']['phishing']:,}",
        f"- Published features: {structural['published_features']}",
        f"- Unique registered domains: {structural['unique_registered_domains']:,}",
        f"- Train rows (train + val): {structural['analysis_splits']['train']:,}",
        f"- Fixed Test rows: {structural['analysis_splits']['test']:,}",
        f"- Train/Test registered-domain overlap: {structural['train_test_registered_domain_overlap']}",
        "",
        "## Train-derived label-conditioned thresholds",
        "",
        f"- URL length threshold: {thresholds['url_more_than_mean']:.6f}",
        f"- Domain length threshold: {thresholds['domain_more_than_mean']:.6f}",
        "",
        "## count_www audit (Train only)",
        "",
        markdown_table(count_www),
        "",
        "## Strongest univariate associations (Train only)",
        "",
        markdown_table(top),
        "",
        "## Interpretation guardrails",
        "",
        "- A high univariate score flags a feature for review; it does not prove leakage.",
        "- A directly visible feature is not automatically invalid.",
        "- The two label-conditioned variables should be recomputed inside each training fold before final modeling.",
        "- Ablation must be pre-specified and reported as robustness analysis rather than selected on Test performance.",
        "",
    ]
    (output / "DATA_UNDERSTANDING_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )


def main() -> None:
    args = arguments()
    output = project_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    frame = read_processed(args.data).copy()
    missing_published = sorted(set(ALL_PUBLISHED_FEATURES) - set(frame.columns))
    if missing_published:
        raise ValueError(
            "Processed CompPhish file does not retain all 70 published features: "
            f"{missing_published}"
        )
    frame["analysis_split"] = two_set_view(frame)

    structural = structural_tables(frame, output)
    train = frame[frame["analysis_split"].eq("train")].reset_index(drop=True)
    associations = label_summary_and_association(train, output, args.seed)
    _, thresholds = label_conditioned_feature_audit(frame, output)
    count_www = count_www_audit(frame, output)

    taxonomy = pd.DataFrame(
        {
            "feature": ALL_PUBLISHED_FEATURES,
            "feature_group": [feature_group(feature) for feature in ALL_PUBLISHED_FEATURES],
            "visibility_tier": [
                visibility_tier(feature) for feature in ALL_PUBLISHED_FEATURES
            ],
        }
    )
    taxonomy.to_csv(
        output / "table_10_feature_visibility_taxonomy.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_label_distribution(frame, output, args.dpi)
    plot_train_test(frame, output, args.dpi)
    plot_feature_groups(output, args.dpi)
    plot_count_www(frame, output, args.dpi)
    plot_top_associations(associations, output, args.top_k, args.dpi)
    plot_length_distributions(frame, output, args.dpi)
    plot_correlation_heatmap(frame, associations, output, args.dpi)
    plot_shortcut_map(associations, output, args.dpi)

    summary = {
        "dataset": "CompPhish v4",
        "analysis_policy": {
            "structural_checks": "complete cleaned corpus",
            "feature_label_association": "Train only (original train + val)",
            "test_used_for_feature_screening": False,
            "feature_removal_decision": "none; tables flag candidates for pre-specified ablation",
        },
        **structural,
        "train_phishing_length_thresholds": thresholds,
        "top_univariate_features_train": associations.head(10)[
            ["feature", "direction_free_roc_auc", "mutual_information", "direction"]
        ].to_dict(orient="records"),
        "count_www_train": count_www.to_dict(orient="records"),
    }
    (output / "data_understanding_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(output, structural, associations, count_www, thresholds)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved Data Understanding package: {output}")
    print("Key report: DATA_UNDERSTANDING_REPORT.md")


if __name__ == "__main__":
    main()
