"""Build a compact runtime comparison figure for the thesis results chapter.

Copyright (c) 2026 Le Thanh Loi. All rights reserved.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


MODEL_ORDER = [
    "full__catboost",
    "full__stacking_boosting",
    "full__stacking_heterogeneous",
]

MODEL_LABELS = {
    "full__catboost": "Full CatBoost",
    "full__stacking_boosting": "Boosting Stacking",
    "full__stacking_heterogeneous": "Heterogeneous Stacking",
}

MODEL_COLORS = {
    "full__catboost": "#3B82B8",
    "full__stacking_boosting": "#F28E2B",
    "full__stacking_heterogeneous": "#D9534F",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="artifacts/compphish/runtime_benchmark_full/runtime_summary.csv",
    )
    parser.add_argument(
        "--output",
        default=(
            "artifacts/compphish/runtime_benchmark_full/"
            "figure_runtime_comparison_representative.png"
        ),
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def add_horizontal_bars(
    ax: plt.Axes,
    data: pd.DataFrame,
    value_column: str,
    xlabel: str,
    panel_title: str,
    value_format: str,
) -> None:
    labels = [MODEL_LABELS[name] for name in data["model"]]
    values = data[value_column].to_numpy(dtype=float)
    colors = [MODEL_COLORS[name] for name in data["model"]]
    bars = ax.barh(labels, values, color=colors, height=0.56)
    ax.invert_yaxis()
    ax.set_title(panel_title, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    largest = max(values)
    ax.set_xlim(0, largest * 1.20)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + largest * 0.018,
            bar.get_y() + bar.get_height() / 2,
            value_format.format(value),
            va="center",
            ha="left",
            fontweight="bold",
        )


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary).resolve()
    output_path = Path(args.output).resolve()

    frame = pd.read_csv(summary_path)
    selected = frame.loc[frame["model"].isin(MODEL_ORDER)].copy()
    selected["model"] = pd.Categorical(
        selected["model"], categories=MODEL_ORDER, ordered=True
    )
    selected = selected.sort_values("model").reset_index(drop=True)
    if len(selected) != len(MODEL_ORDER):
        found = set(selected["model"].astype(str))
        missing = [name for name in MODEL_ORDER if name not in found]
        raise ValueError(f"Missing benchmark rows: {missing}")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    fig.suptitle(
        "So sánh chi phí tính toán của các mô hình đại diện trên nhóm Full",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )

    add_horizontal_bars(
        axes[0],
        selected,
        "median_training_seconds",
        "Thời gian huấn luyện trung vị (giây)",
        "(a) Huấn luyện",
        "{:.2f} giây",
    )
    add_horizontal_bars(
        axes[1],
        selected,
        "median_inference_ms_per_sample",
        "Thời gian suy luận trung vị (ms/mẫu)",
        "(b) Suy luận",
        "{:.4f} ms",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=3.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
