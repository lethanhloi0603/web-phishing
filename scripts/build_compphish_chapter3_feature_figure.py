# Copyright (c) 2026 Le Thanh Loi. All rights reserved.
# Author: Le Thanh Loi

"""Create the Chapter 3 figure describing CompPhish feature structure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="artifacts/compphish/chapter3_figures",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })

    groups = ["Brand-jacking", "URL-based", "HTML"]
    totals = np.array([5, 42, 23])
    boolean = np.array([5, 14, 14])
    count_based = np.array([0, 20, 0])
    integer = np.array([0, 4, 8])
    real_valued = np.array([0, 4, 1])

    colors = {
        "group": ["#7A5195", "#2F6B9A", "#4C956C"],
        "Boolean": "#3977A8",
        "Số đếm": "#E68632",
        "Số nguyên": "#4C956C",
        "Số thực": "#8B6BB1",
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 4.8),
        gridspec_kw={"width_ratios": [1.0, 1.25]},
    )

    y = np.arange(len(groups))
    bars = axes[0].barh(y, totals, color=colors["group"], height=0.56)
    axes[0].set_yticks(y, labels=groups)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 46)
    axes[0].set_xlabel("Số lượng đặc trưng")
    axes[0].set_title("(a) Số lượng theo nguồn thông tin", pad=12)
    axes[0].grid(axis="x", alpha=0.24, linewidth=0.8)
    axes[0].set_axisbelow(True)
    for bar, value in zip(bars, totals):
        axes[0].text(
            bar.get_width() + 0.7,
            bar.get_y() + bar.get_height() / 2,
            f"{value}",
            va="center",
            ha="left",
            fontweight="bold",
        )

    left = np.zeros(len(groups))
    series = [
        ("Boolean", boolean),
        ("Số đếm", count_based),
        ("Số nguyên", integer),
        ("Số thực", real_valued),
    ]
    for label, values in series:
        segment = axes[1].barh(
            y,
            values,
            left=left,
            height=0.56,
            label=label,
            color=colors[label],
        )
        for bar, value in zip(segment, values):
            if value <= 0:
                continue
            x = bar.get_x() + bar.get_width() / 2
            axes[1].text(
                x,
                bar.get_y() + bar.get_height() / 2,
                str(int(value)),
                va="center",
                ha="center",
                color="white" if value >= 4 else "black",
                fontweight="bold",
            )
        left += values

    axes[1].set_yticks(y, labels=groups)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 46)
    axes[1].set_xlabel("Số lượng đặc trưng")
    axes[1].set_title("(b) Kiểu dữ liệu trong từng nhóm", pad=12)
    axes[1].grid(axis="x", alpha=0.24, linewidth=0.8)
    axes[1].set_axisbelow(True)
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        frameon=False,
    )

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.suptitle(
        "Cấu trúc và kiểu dữ liệu của 70 đặc trưng CompPhish",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout(w_pad=3.0)

    png_path = output_dir / "figure_03_feature_structure.png"
    pdf_path = output_dir / "figure_03_feature_structure.pdf"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {png_path.resolve()}")
    print(f"Saved: {pdf_path.resolve()}")


if __name__ == "__main__":
    main()
