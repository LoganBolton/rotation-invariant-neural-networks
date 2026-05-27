"""Plot sweep results saved from sweep_kchain.py markdown logs."""

from __future__ import annotations

import argparse
import ast
import os
import re
from pathlib import Path

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib-cache")))

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

RESULT_RE = re.compile(
    r"^\s*(?P<successes>\d+)/(?P<trials>\d+)\s*\|\s*"
    r"(?P<k>\d+)\s*\|\s*"
    r"(?P<cutoff>[-+]?\d+(?:\.\d+)?)\s*\|\s*"
    r"(?P<layers>\d+)\s*\|\s*"
    r"(?P<accuracies>\[[^\]]*\])\s*\|\s*"
    r"(?P<margin_accuracies>\[[^\]]*\])\s*\|"
)
PARAM_RE = re.compile(r"Using params l-max:\s*(?P<l_max>\d+)\s+and n-max:\s*(?P<n_max>\d+)")

K_GEOMETRY = {
    2: {"max_diameter": 13.60},
    3: {"max_diameter": 17.89},
    4: {"max_diameter": 22.47},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_file", type=Path, help="Markdown file containing sweep_kchain.py output.")
    parser.add_argument("--output", type=Path, default=None, help="Output image path.")
    parser.add_argument("--min-layers", type=int, default=1, help="Minimum interaction-layer count to display.")
    parser.add_argument(
        "--metric",
        choices=["accuracy", "margin_accuracy", "success_rate"],
        default="accuracy",
        help="Value to color and annotate in each grid cell.",
    )
    return parser.parse_args()


def parse_model_label(log_file: Path) -> str:
    for line in log_file.read_text().splitlines():
        match = PARAM_RE.search(line)
        if match is None:
            continue

        l_max = int(match.group("l_max"))
        n_max = int(match.group("n_max"))
        if l_max == 0 and n_max == 1:
            return "HIP-NN"
        return f"HIP-HOP-NN (l={l_max}, n={n_max})"

    return log_file.stem


def parse_results(log_file: Path) -> list[dict[str, float]]:
    rows = []
    for line in log_file.read_text().splitlines():
        match = RESULT_RE.match(line)
        if match is None:
            continue

        accuracies = ast.literal_eval(match.group("accuracies"))
        margin_accuracies = ast.literal_eval(match.group("margin_accuracies"))
        successes = int(match.group("successes"))
        trials = int(match.group("trials"))

        rows.append(
            {
                "k": int(match.group("k")),
                "cutoff": float(match.group("cutoff")),
                "layers": int(match.group("layers")),
                "accuracy": float(np.mean(accuracies)),
                "margin_accuracy": float(np.mean(margin_accuracies)),
                "success_rate": successes / trials,
            }
        )

    if not rows:
        raise ValueError(f"No sweep result rows found in {log_file}.")
    return rows


def plot_results(rows: list[dict[str, float]], metric: str, output_file: Path, min_layers: int, model_label: str) -> None:
    rows = [row for row in rows if row["layers"] >= min_layers]
    if not rows:
        raise ValueError(f"No sweep result rows remain after filtering to layers >= {min_layers}.")

    ks = sorted({row["k"] for row in rows})
    cutoffs = sorted({row["cutoff"] for row in rows})
    layers = sorted({row["layers"] for row in rows})

    fig, axes = plt.subplots(1, len(ks), figsize=(4.0 * len(ks), 3.8), constrained_layout=True, squeeze=False)
    cmap = mcolors.LinearSegmentedColormap.from_list("failure_to_success", ["#c62828", "#f7f7f7", "#2e7d32"])

    for axis, k in zip(axes[0], ks):
        grid = np.full((len(cutoffs), len(layers)), np.nan)
        for row in rows:
            if row["k"] != k:
                continue
            y = cutoffs.index(row["cutoff"])
            x = layers.index(row["layers"])
            grid[y, x] = row[metric]

        image = axis.imshow(grid, cmap=cmap, vmin=0.5, vmax=1.0, origin="lower", aspect="auto")
        geometry = K_GEOMETRY.get(k)
        if geometry is None:
            axis.set_title(f"k={k}")
        else:
            axis.set_title(f"k={k}\nmax diameter: {geometry['max_diameter']:.2f}")
        axis.set_xlabel("interaction layers")
        axis.set_xticks(range(len(layers)), layers)
        axis.set_yticks(range(len(cutoffs)), [f"{cutoff:g}" for cutoff in cutoffs])
        axis.set_ylabel("hard cutoff")

        for y, cutoff in enumerate(cutoffs):
            for x, layer in enumerate(layers):
                value = grid[y, x]
                if np.isnan(value):
                    continue
                text_color = "white" if value < 0.65 or value > 0.9 else "black"
                axis.text(x, y, f"{value:.2f}", ha="center", va="center", color=text_color, fontweight="bold")

    fig.suptitle(f"{model_label} {metric.replace('_', ' ')}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_file = args.output
    if output_file is None:
        output_file = args.log_file.with_name(f"{args.log_file.stem}_{args.metric}_grid.png")

    rows = parse_results(args.log_file)
    model_label = parse_model_label(args.log_file)
    plot_results(rows, args.metric, output_file, args.min_layers, model_label)
    print(f"Saved plot: {output_file}")


if __name__ == "__main__":
    main()
