"""Make the sectioned l_max=3/l_max=4 margin-accuracy paper figure.

This is an alternate presentation of ``plot_paper_margin_comparison.py``. It
uses one heatmap, square cells, a strong divider between the two l_max
sections, and only the ten unique graph configurations.

Run from the repository root:

    python benchmarks/rotating_ring/plot_paper_margin_comparison_sectioned.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle

import plot_paper_margin_comparison as base


DEFAULT_OUTPUT = (
    base.DEFAULT_RESULTS_DIR / "paper_margin_l3_l4_sectioned_vs_equiformer"
)
DISPLAY_GRAPH_KEYS = base.GRAPH_KEYS[2:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--equiformer",
        type=Path,
        default=base.DEFAULT_EQUIFORMER,
        help="Equiformer JSON result file.",
    )
    parser.add_argument(
        "--hiphop-l3",
        type=Path,
        default=base.DEFAULT_HIPHOP_L3,
        help="HIP-HOP result directory containing l3_n4.json and l3_n5.json.",
    )
    parser.add_argument(
        "--hiphop-l4",
        type=Path,
        default=base.DEFAULT_HIPHOP_L4,
        help="HIP-HOP result directory containing l4_n4.json.",
    )
    parser.add_argument(
        "--hiphop-l4-n7",
        type=Path,
        default=base.DEFAULT_HIPHOP_L4_N7,
        help="HIP-HOP result directory containing l4_n7.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path without a suffix. Both PNG and PDF are written.",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Figure title. Pass an empty string to omit it.",
    )
    return parser.parse_args()


def plot_figure(
    equiformer_path: Path,
    hiphop_l3_root: Path,
    hiphop_l4_root: Path,
    hiphop_l4_n7_root: Path,
    output: Path,
    title: str,
) -> None:
    sections = [
        (
            4,
            [
                base.MarginAccuracyRow(
                    r"HIP-HOP ($n=7$)",
                    base.load_hiphop(hiphop_l4_n7_root, 4, 7),
                ),
                base.MarginAccuracyRow(
                    r"HIP-HOP ($n=4$)",
                    base.load_hiphop(hiphop_l4_root, 4, 4),
                ),
                base.MarginAccuracyRow(
                    "EquiformerV3",
                    base.load_equiformer(equiformer_path, 4),
                ),
            ],
            6,
        ),
        (
            3,
            [
                base.MarginAccuracyRow(
                    r"HIP-HOP ($n=5$)",
                    base.load_hiphop(hiphop_l3_root, 3, 5),
                ),
                base.MarginAccuracyRow(
                    r"HIP-HOP ($n=4$)",
                    base.load_hiphop(hiphop_l3_root, 3, 4),
                ),
                base.MarginAccuracyRow(
                    "EquiformerV3",
                    base.load_equiformer(equiformer_path, 3),
                ),
            ],
            3,
        ),
    ]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "font.weight": "bold",
            "mathtext.fontset": "stix",
            "mathtext.default": "bf",
            "font.size": 10,
            "text.color": "#202020",
            "axes.labelcolor": "#202020",
            "axes.edgecolor": "#303030",
            "axes.labelsize": 11,
            "axes.labelweight": "bold",
            "xtick.color": "#202020",
            "ytick.color": "#202020",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.4, 5.1),
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    # A true red makes failures more immediate than the orange used previously.
    cmap = ListedColormap(["#C83E3B", "#278C5A"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    # Group the two HIP-HOP cells at the configurations being contrasted.
    highlight_color = "#111111"
    highlight_width = 0.075

    def add_inner_border(
        ax: plt.Axes,
        x_left: float,
        y_top: float,
        x_right: float,
        y_bottom: float,
    ) -> None:
        strips = (
            (x_left, y_top, highlight_width, y_bottom - y_top),
            (x_right - highlight_width, y_top, highlight_width, y_bottom - y_top),
            (x_left, y_top, x_right - x_left, highlight_width),
            (x_left, y_bottom - highlight_width, x_right - x_left, highlight_width),
        )
        for x, y, width, height in strips:
            ax.add_patch(
                Rectangle(
                    (x, y),
                    width,
                    height,
                    facecolor=highlight_color,
                    edgecolor="none",
                    zorder=6,
                )
            )

    image = None
    for section_index, (l_max, rows, highlight_column) in enumerate(sections):
        ax = axes[section_index]
        matrix = np.asarray([row.values[2:] for row in rows])
        image = ax.imshow(
            matrix,
            cmap=cmap,
            norm=norm,
            aspect="equal",
            interpolation="none",
        )
        ax.set_anchor("S" if section_index == 0 else "N")

        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    color="#111111",
                    fontsize=9,
                    fontweight="bold",
                )

        ax.set_yticks(range(len(rows)), [row.label for row in rows])
        ax.tick_params(axis="y", length=0, pad=9, labelsize=11)
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")
        ax.set_xticks(
            range(len(DISPLAY_GRAPH_KEYS)),
            [f"{n_inner},{n_outer}" for n_inner, n_outer in DISPLAY_GRAPH_KEYS],
        )
        ax.tick_params(
            axis="x",
            bottom=section_index == 1,
            labelbottom=section_index == 1,
            length=0,
            pad=6,
            labelsize=10,
        )
        for label in ax.get_xticklabels():
            label.set_fontweight("bold")
        ax.set_xticks(np.arange(-0.5, len(DISPLAY_GRAPH_KEYS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.9, alpha=0.95)
        ax.tick_params(which="minor", bottom=False, left=False)

        add_inner_border(
            ax,
            highlight_column - 0.5,
            -0.5,
            highlight_column + 0.5,
            1.5,
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
        section_box_x = -0.31
        section_box_width = 0.06
        ax.add_patch(
            Rectangle(
                (section_box_x, 0.0),
                section_box_width,
                1.0,
                transform=ax.transAxes,
                facecolor="#E8E8E8",
                edgecolor="#303030",
                linewidth=1.2,
                clip_on=False,
                zorder=7,
            )
        )
        ax.text(
            section_box_x + section_box_width / 2,
            0.5,
            rf"$\ell_{{\max}}={l_max}$",
            transform=ax.transAxes,
            ha="center",
            va="center",
            rotation=90,
            fontsize=12,
            fontweight="bold",
            zorder=8,
        )

    if title:
        axes[0].set_title(title, fontsize=15, fontweight="normal", pad=14)
    axes[1].set_xlabel(
        "Graph Configuration (# Inner Rings, # Outer Rings)",
        labelpad=8,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.27, right=0.98, bottom=0.14, top=0.93, hspace=0.08)
    output.parent.mkdir(parents=True, exist_ok=True)
    png_path = output.with_suffix(".png")
    pdf_path = output.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def main() -> None:
    args = parse_args()
    plot_figure(
        args.equiformer,
        args.hiphop_l3,
        args.hiphop_l4,
        args.hiphop_l4_n7,
        args.output,
        args.title,
    )


if __name__ == "__main__":
    main()
