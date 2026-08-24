import math

import pandas as pd


def _label_series(frame):
    return pd.to_numeric(frame["label"], errors="coerce").fillna(-1).astype(int)


def source_group(frame):
    """Map heterogeneous provenance fields to the paper's Human/LLM axis."""
    result = pd.Series("human", index=frame.index, dtype="object")
    if "source" in frame:
        values = frame["source"].fillna("").astype(str).str.lower()
        result.loc[values.str.contains("llm|model|generated", regex=True)] = "llm"
    if "dataset_domain" in frame:
        values = frame["dataset_domain"].fillna("").astype(str).str.lower()
        result.loc[values.str.contains("llm|synthetic", regex=True)] = "llm"
    return result


def contingency(frame, column):
    if column not in frame:
        return {}
    work = pd.DataFrame({"group": frame[column].fillna("unknown").astype(str), "label": _label_series(frame)})
    work = work[work["label"].isin([0, 1])]
    table = pd.crosstab(work["group"], work["label"])
    report = {}
    for group, row in table.iterrows():
        negative = int(row.get(0, 0))
        positive = int(row.get(1, 0))
        total = negative + positive
        report[str(group)] = {
            "negative": negative,
            "positive": positive,
            "total": total,
            "label_purity": max(negative, positive) / total if total else math.nan,
        }
    return report


def shortcut_report(frame, min_group_size=20, max_group_purity=0.95):
    work = frame.copy()
    work["source_group"] = source_group(work)
    source = contingency(work, "source_group")
    missing_source_label_cells = [
        f"{group}/label={label}"
        for group in ("human", "llm")
        for label in (0, 1)
        if source.get(group, {}).get("negative" if label == 0 else "positive", 0) == 0
    ]

    grouped = {
        column: contingency(work, column)
        for column in ("source_group", "brand", "generator_family", "generator")
        if column in work
    }
    high_purity = {}
    for column, groups in grouped.items():
        flagged = {
            name: values
            for name, values in groups.items()
            if values["total"] >= min_group_size
            and values["label_purity"] > max_group_purity
        }
        if flagged:
            high_purity[column] = flagged

    return {
        "rows": int(len(work)),
        "source_label_2x2": source,
        "missing_source_label_cells": missing_source_label_cells,
        "groups": grouped,
        "high_purity_groups": high_purity,
        "settings": {
            "min_group_size": int(min_group_size),
            "max_group_purity": float(max_group_purity),
        },
    }
