"""Make the paper figure comparing Equiformer and selected HIP-HOP configs.

The default inputs reproduce the constant-radius comparison for:

* Equiformer at l_max = 4
* HIP-HOP at (l_max, n_max) = (3, 5), (4, 4), and (4, 7)

Only graph configurations with N_inner <= N_outer are shown because exchanging
the inner and outer node counts produces an equivalent configuration. The
figure reports margin accuracy and orders rows from lower to higher aggregate
accuracy as the y axis increases.

Run from the repository root:

    python benchmarks/rotating_ring/plot_paper_margin_comparison.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


DEFAULT_RESULTS_DIR = Path(
    "benchmarks/rotating_ring/results/model_comparisons/"
    "equi_vs_hiphop_const_radius"
)
DEFAULT_EQUIFORMER = DEFAULT_RESULTS_DIR / "equiformer.json"
DEFAULT_HIPHOP_L3 = Path(
    "benchmarks/rotating_ring/results/2d_l2_l3_n2to5_gap0/json_results"
)
DEFAULT_HIPHOP_L4 = Path(
    "benchmarks/rotating_ring/results/"
    "2d_l3_l4_n3to5_cutoff7_equal_radius_local/json_results"
)
DEFAULT_HIPHOP_L4_N7 = Path(
    "benchmarks/rotating_ring/results/2d_l4_n7_gap0/json_results"
)
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "paper_margin_config_stack_vs_equiformer"

GRAPH_KEYS = [
    (n_inner, n_outer)
    for n_inner in range(1, 5)
    for n_outer in range(n_inner, 5)
]


@dataclass(frozen=True)
class MarginAccuracyRow:
    label: str
    values: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--equiformer",
        type=Path,
        default=DEFAULT_EQUIFORMER,
        help="Equiformer JSON result file.",
    )
    parser.add_argument(
        "--hiphop-l3",
        type=Path,
        default=DEFAULT_HIPHOP_L3,
        help="HIP-HOP result directory containing l3_n5.json.",
    )
    parser.add_argument(
        "--hiphop-l4",
        type=Path,
        default=DEFAULT_HIPHOP_L4,
        help="HIP-HOP result directory containing l4_n4.json.",
    )
    parser.add_argument(
        "--hiphop-l4-n7",
        type=Path,
        default=DEFAULT_HIPHOP_L4_N7,
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
        default="Rotating-ring margin accuracy",
        help="Figure title. Pass an empty string to omit it.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def nested_get(record: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value: Any = record
        for key in path:
            if not isinstance(value, dict) or key not in value:
                break
            value = value[key]
        else:
            return value
    return None


def record_values(record: dict[str, Any]) -> tuple[tuple[int, int], int, int | None, float]:
    n_inner = nested_get(
        record,
        ("config", "graph", "ring_n_inner"),
        ("config", "ring_n_inner"),
    )
    n_outer = nested_get(
        record,
        ("config", "graph", "ring_n_outer"),
        ("config", "ring_n_outer"),
    )
    l_max = nested_get(
        record,
        ("config", "model", "lmax"),
        ("config", "model", "l_max"),
        ("config", "l_max"),
    )
    n_max = nested_get(
        record,
        ("config", "model", "nmax"),
        ("config", "model", "n_max"),
        ("config", "n_max"),
    )
    margin_accuracy = nested_get(
        record,
        ("metrics", "mean_margin_accuracy"),
        ("metrics", "margin_accuracy"),
        ("margin_accuracy",),
    )
    if n_inner is None or n_outer is None or l_max is None or margin_accuracy is None:
        raise ValueError("Result record is missing graph, l_max, or margin accuracy.")
    return (
        (int(n_inner), int(n_outer)),
        int(l_max),
        None if n_max is None else int(n_max),
        float(margin_accuracy),
    )


def ordered_values(values: dict[tuple[int, int], float], description: str) -> list[float]:
    missing = [key for key in GRAPH_KEYS if key not in values]
    extra = [key for key in values if key not in GRAPH_KEYS]
    if missing or extra:
        raise ValueError(f"{description} has missing graph keys {missing} and extra graph keys {extra}.")
    return [values[key] for key in GRAPH_KEYS]


def load_equiformer(path: Path, l_max: int) -> list[float]:
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")

    values: dict[tuple[int, int], float] = {}
    for record in data:
        graph_key, record_l_max, _, margin_accuracy = record_values(record)
        if record_l_max != l_max:
            continue
        if graph_key not in GRAPH_KEYS:
            continue
        if graph_key in values:
            raise ValueError(f"Duplicate Equiformer result for l_max={l_max}, graph={graph_key}.")
        values[graph_key] = margin_accuracy
    return ordered_values(values, f"Equiformer l_max={l_max}")


def load_hiphop(root: Path, l_max: int, n_max: int) -> list[float]:
    values: dict[tuple[int, int], float] = {}
    result_name = f"l{l_max}_n{n_max}.json"
    for path in sorted(root.glob(f"2d_*inner_*_outer/{result_name}")):
        record = read_json(path)
        if not isinstance(record, dict):
            raise ValueError(f"{path} must contain one JSON object.")
        graph_key, record_l_max, record_n_max, margin_accuracy = record_values(record)
        if (record_l_max, record_n_max) != (l_max, n_max):
            raise ValueError(
                f"{path} contains (l_max, n_max)=({record_l_max}, {record_n_max}), "
                f"expected ({l_max}, {n_max})."
            )
        if graph_key not in GRAPH_KEYS:
            continue
        if graph_key in values:
            raise ValueError(f"Duplicate HIP-HOP result for graph={graph_key} in {root}.")
        values[graph_key] = margin_accuracy
    return ordered_values(values, f"HIP-HOP l_max={l_max}, n_max={n_max}")


def annotation_color(value: float) -> str:
    return "white" if value < 0.5 else "#111111"


def draw_heatmap(
    ax: plt.Axes,
    rows: list[MarginAccuracyRow],
    norm: BoundaryNorm,
    cmap: ListedColormap,
) -> matplotlib.image.AxesImage:
    matrix = np.asarray([row.values for row in rows])
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="none")

    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            ax.text(
                column_index,
                row_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                color=annotation_color(value),
                fontsize=8.0,
                fontweight="bold",
            )

    ax.set_yticks(range(len(rows)), [row.label for row in rows])
    ax.tick_params(axis="y", length=0, pad=7, labelsize=9)

    ax.set_xticks(range(len(GRAPH_KEYS)), [f"{n_inner}{n_outer}" for n_inner, n_outer in GRAPH_KEYS])
    ax.tick_params(
        axis="x",
        length=0,
        pad=4,
        labelsize=8.5,
    )
    ax.set_xlabel(
        "Graph Configuration (# Inner Rings, # Outer Rings)",
        labelpad=7,
    )

    ax.set_xticks(np.arange(-0.5, len(GRAPH_KEYS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.75, alpha=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)

    for boundary in (3.5, 6.5, 8.5):
        ax.axvline(boundary, color="#202020", linewidth=1.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#303030")
    return image


def plot_figure(
    equiformer_path: Path,
    hiphop_l3_root: Path,
    hiphop_l4_root: Path,
    hiphop_l4_n7_root: Path,
    output: Path,
    title: str,
) -> None:
    bottom_to_top = [
        MarginAccuracyRow(
            r"HIP-HOP ($\ell_{\max}=3,\ n_{\max}=5$)",
            load_hiphop(hiphop_l3_root, 3, 5),
        ),
        MarginAccuracyRow(
            r"HIP-HOP ($\ell_{\max}=4,\ n_{\max}=4$)",
            load_hiphop(hiphop_l4_root, 4, 4),
        ),
        MarginAccuracyRow(
            r"Equiformer ($\ell_{\max}=4$)",
            load_equiformer(equiformer_path, 4),
        ),
        MarginAccuracyRow(
            r"HIP-HOP ($\ell_{\max}=4,\ n_{\max}=7$)",
            load_hiphop(hiphop_l4_n7_root, 4, 7),
        ),
    ]
    row_means = [float(np.mean(row.values)) for row in bottom_to_top]
    if row_means != sorted(row_means):
        raise ValueError(
            "Rows are not ordered from lower to higher aggregate margin accuracy: "
            f"{row_means}"
        )
    # imshow places its first row at the top, so reverse the desired y-axis order.
    rows = list(reversed(bottom_to_top))

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(8.1, 3.5), constrained_layout=True)
    cmap = ListedColormap(["#D55E00", "#009E73"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    image = draw_heatmap(ax, rows, norm, cmap)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.018)
    colorbar.set_label("Margin accuracy")
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["0.0", "1.0"])
    colorbar.outline.set_linewidth(0.7)

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
