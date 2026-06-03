"""Plot sweep results saved from sweep.py markdown logs."""

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
    r"(?P<counterexample>[A-Za-z0-9_ -]+?)\s*\|\s*"
    r"(?P<cutoff>[-+]?\d+(?:\.\d+)?)\s*\|\s*"
    r"(?P<layers>\d+)\s*\|\s*"
    r"(?P<accuracies>\[[^\]]*\])\s*\|\s*"
    r"(?P<margin_accuracies>\[[^\]]*\])\s*\|"
)

PARAM_RE = re.compile(r"Using params l-max:\s*(?P<l_max>\d+)\s+and n-max:\s*(?P<n_max>\d+)")

COUNTEREXAMPLE_GEOMETRY = {
    "two_body": {"max_diameter": 10.0},
    "three_body": {"max_diameter": 14.14},
    "four_body_nonchiral": {"max_diameter": 10.66},
    "four_body_chiral": {"max_diameter": 10.0},
}

COUNTEREXAMPLE_ORDER = (
    "two_body",
    "three_body",
    "four_body_nonchiral",
    "four_body_chiral",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path, help="Markdown sweep log or directory containing sweep logs.")
    parser.add_argument("--output", type=Path, default=None, help="Output image path for a single-log plot.")
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=None,
        help="Output image path for the shared multi-model plot.",
    )
    parser.add_argument("--min-layers", type=int, default=1, help="Minimum interaction-layer count to display.")
    parser.add_argument(
        "--average-cutoff",
        "--average-cutoffs",
        action="store_true",
        dest="average_cutoffs",
        help="Average results over all hard cutoffs and show one cutoff row.",
    )
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
        return f"HIP-HOP-NN\nl={l_max}, n={n_max}"

    return log_file.stem


def model_sort_key(log_file: Path) -> tuple[int, int, str]:
    match = re.fullmatch(r"l(?P<l_max>\d+)_n(?P<n_max>\d+)", log_file.stem)
    if match is None:
        return (10_000, 10_000, log_file.stem)
    return (int(match.group("n_max")), int(match.group("l_max")), log_file.stem)


def result_log_files(result_dir: Path) -> list[Path]:
    log_files = sorted(result_dir.glob("*.md"), key=model_sort_key)
    if not log_files:
        raise ValueError(f"No markdown sweep logs found in {result_dir}.")
    return log_files


def output_metric_name(metric: str, average_cutoffs: bool) -> str:
    if average_cutoffs:
        return f"{metric}_average_cutoffs"
    return metric


def parse_results(log_file: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []

    for line in log_file.read_text().splitlines():
        match = RESULT_RE.match(line)
        if match is None:
            continue

        accuracies = ast.literal_eval(match.group("accuracies"))
        margin_accuracies = ast.literal_eval(match.group("margin_accuracies"))
        successes = int(match.group("successes"))
        trials = int(match.group("trials"))
        success_rate = successes / trials

        rows.append(
            {
                "counterexample": match.group("counterexample").strip(),
                "cutoff": float(match.group("cutoff")),
                "layers": int(match.group("layers")),
                "accuracy": success_rate,
                "margin_accuracy": float(np.mean(margin_accuracies)),
                "success_rate": success_rate,
            }
        )

    if not rows:
        raise ValueError(f"No sweep result rows found in {log_file}.")

    return rows


def ordered_items(rows_by_model: list[tuple[str, list[dict[str, float | str]]]]) -> list[str]:
    item_set = {
        str(row["counterexample"])
        for _model_label, rows in rows_by_model
        for row in rows
    }
    ordered = [
        counterexample
        for counterexample in COUNTEREXAMPLE_ORDER
        if counterexample in item_set
    ]

    remaining = item_set - set(ordered)
    numeric = sorted(
        (item for item in remaining if item.isdigit()),
        key=lambda item: int(item),
    )
    text = sorted(remaining - set(numeric))
    return ordered + numeric + text


def item_title(item: str) -> str:
    geometry = COUNTEREXAMPLE_GEOMETRY.get(item)
    title = item.replace("_", " ")
    if geometry is not None:
        title += f"\nmax diameter: {geometry['max_diameter']:.2f}"
    elif item.isdigit():
        title = f"k={item}"
    return title


def plot_results(
    rows: list[dict[str, float | str]],
    metric: str,
    output_file: Path,
    min_layers: int,
    model_label: str,
    average_cutoffs: bool,
) -> None:
    rows = [row for row in rows if int(row["layers"]) >= min_layers]
    if not rows:
        raise ValueError(f"No sweep result rows remain after filtering to layers >= {min_layers}.")

    counterexamples = ordered_items([(model_label, rows)])
    cutoffs = ["average"] if average_cutoffs else sorted({float(row["cutoff"]) for row in rows})
    layers = sorted({int(row["layers"]) for row in rows})

    fig_width = max(4.0 * len(counterexamples), 6.0)
    fig, axes = plt.subplots(
        1,
        len(counterexamples),
        figsize=(fig_width, 3.8),
        constrained_layout=True,
        squeeze=False,
    )

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "failure_to_success",
        ["#c62828", "#f7f7f7", "#2e7d32"],
    )

    for axis, counterexample in zip(axes[0], counterexamples):
        grid = np.full((len(cutoffs), len(layers)), np.nan)

        for x, layer in enumerate(layers):
            if average_cutoffs:
                values = [
                    float(row[metric])
                    for row in rows
                    if str(row["counterexample"]) == counterexample
                    and int(row["layers"]) == layer
                ]
                if values:
                    grid[0, x] = float(np.mean(values))
                continue

            for row in rows:
                if str(row["counterexample"]) != counterexample:
                    continue
                if int(row["layers"]) != layer:
                    continue

                y = cutoffs.index(float(row["cutoff"]))
                grid[y, x] = float(row[metric])

        image = axis.imshow(
            grid,
            cmap=cmap,
            vmin=0.0 if metric in {"accuracy", "success_rate"} else 0.5,
            vmax=1.0,
            origin="lower",
            aspect="equal" if average_cutoffs else "auto",
        )

        axis.set_title(item_title(counterexample))

        axis.set_xlabel("interaction layers")
        axis.set_xticks(range(len(layers)), layers)
        if average_cutoffs:
            axis.set_yticks([])
            axis.set_ylabel("")
        else:
            axis.set_yticks(range(len(cutoffs)), [f"{cutoff:g}" for cutoff in cutoffs])
            axis.set_ylabel("hard cutoff")
        axis.set_xticks(np.arange(len(layers) + 1) - 0.5, minor=True)
        axis.set_yticks(np.arange(len(cutoffs) + 1) - 0.5, minor=True)
        axis.grid(which="minor", color="black", linewidth=1.0)
        axis.tick_params(which="minor", bottom=False, left=False)

        for y, _cutoff in enumerate(cutoffs):
            for x, _layer in enumerate(layers):
                value = grid[y, x]
                if np.isnan(value):
                    continue

                text_color = "white" if value < 0.35 or value > 0.85 else "black"
                axis.text(
                    x,
                    y,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight="bold",
                )

    fig.suptitle(f"{model_label} {metric.replace('_', ' ')}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def plot_combined_results(
    rows_by_model: list[tuple[str, list[dict[str, float | str]]]],
    metric: str,
    output_file: Path,
    min_layers: int,
    average_cutoffs: bool,
) -> None:
    filtered_rows_by_model = [
        (model_label, [row for row in rows if int(row["layers"]) >= min_layers])
        for model_label, rows in rows_by_model
    ]
    filtered_rows_by_model = [
        (model_label, rows)
        for model_label, rows in filtered_rows_by_model
        if rows
    ]
    if not filtered_rows_by_model:
        raise ValueError(f"No sweep result rows remain after filtering to layers >= {min_layers}.")

    items = ordered_items(filtered_rows_by_model)
    cutoffs = (
        ["average"]
        if average_cutoffs
        else sorted({
            float(row["cutoff"])
            for _model_label, rows in filtered_rows_by_model
            for row in rows
        })
    )
    layers = sorted({
        int(row["layers"])
        for _model_label, rows in filtered_rows_by_model
        for row in rows
    })

    n_models = len(filtered_rows_by_model)
    n_items = len(items)
    fig_width = max(2.8 * n_items + 1.8, 7.0)
    fig_height = max((0.9 if average_cutoffs else 2.15) * n_models + 1.0, 3.2)
    fig, axes = plt.subplots(
        n_models,
        n_items,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        gridspec_kw={"hspace": 0.04 if average_cutoffs else 0.18},
        squeeze=False,
    )

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "failure_to_success",
        ["#c62828", "#f7f7f7", "#2e7d32"],
    )
    image = None

    for y_model, (model_label, rows) in enumerate(filtered_rows_by_model):
        for x_item, item in enumerate(items):
            axis = axes[y_model, x_item]
            grid = np.full((len(cutoffs), len(layers)), np.nan)

            for x, layer in enumerate(layers):
                if average_cutoffs:
                    values = [
                        float(row[metric])
                        for row in rows
                        if str(row["counterexample"]) == item
                        and int(row["layers"]) == layer
                    ]
                    if values:
                        grid[0, x] = float(np.mean(values))
                    continue

                for row in rows:
                    if str(row["counterexample"]) != item:
                        continue
                    if int(row["layers"]) != layer:
                        continue
                    y = cutoffs.index(float(row["cutoff"]))
                    grid[y, x] = float(row[metric])

            image = axis.imshow(
                grid,
                cmap=cmap,
                vmin=0.0 if metric in {"accuracy", "success_rate"} else 0.5,
                vmax=1.0,
                origin="lower",
                aspect="equal" if average_cutoffs else "auto",
            )

            if y_model == 0:
                axis.set_title(item_title(item))
            if x_item == 0:
                axis.set_ylabel(model_label if average_cutoffs else f"{model_label}\ncutoff")
            else:
                axis.set_ylabel("")

            axis.set_xticks(range(len(layers)), layers)
            if average_cutoffs:
                axis.set_yticks([])
            else:
                axis.set_yticks(range(len(cutoffs)), [f"{cutoff:g}" for cutoff in cutoffs])
            axis.tick_params(axis="both", labelsize=8)
            axis.set_xticks(np.arange(len(layers) + 1) - 0.5, minor=True)
            axis.set_yticks(np.arange(len(cutoffs) + 1) - 0.5, minor=True)
            axis.grid(which="minor", color="black", linewidth=1.0)
            axis.tick_params(which="minor", bottom=False, left=False)

            if y_model == n_models - 1:
                axis.set_xlabel("layers")
            else:
                axis.set_xlabel("")
                axis.tick_params(labelbottom=False)

            for y, _cutoff in enumerate(cutoffs):
                for x, _layer in enumerate(layers):
                    value = grid[y, x]
                    if np.isnan(value):
                        continue

                    text_color = "white" if value < 0.35 or value > 0.85 else "black"
                    axis.text(
                        x,
                        y,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=8,
                        fontweight="bold",
                    )

    fig.suptitle(f"Model comparison: {metric.replace('_', ' ')}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.input_path.is_dir():
        log_files = result_log_files(args.input_path)
        combined_output = args.combined_output
        if combined_output is None:
            combined_output = args.input_path / f"combined_{output_metric_name(args.metric, args.average_cutoffs)}_grid.png"
        rows_by_model = [(parse_model_label(log_file), parse_results(log_file)) for log_file in log_files]
        plot_combined_results(rows_by_model, args.metric, combined_output, args.min_layers, args.average_cutoffs)
        print(f"Saved combined plot: {combined_output}")
        return

    output_file = args.output
    if output_file is None:
        output_file = args.input_path.with_name(
            f"{args.input_path.stem}_{output_metric_name(args.metric, args.average_cutoffs)}_grid.png"
        )

    rows = parse_results(args.input_path)
    model_label = parse_model_label(args.input_path)
    plot_results(rows, args.metric, output_file, args.min_layers, model_label, args.average_cutoffs)
    print(f"Saved plot: {output_file}")

    sibling_logs = result_log_files(args.input_path.parent)
    if len(sibling_logs) > 1:
        combined_output = args.combined_output
        if combined_output is None:
            combined_output = args.input_path.parent / (
                f"combined_{output_metric_name(args.metric, args.average_cutoffs)}_grid.png"
            )
        rows_by_model = [(parse_model_label(log_file), parse_results(log_file)) for log_file in sibling_logs]
        plot_combined_results(rows_by_model, args.metric, combined_output, args.min_layers, args.average_cutoffs)
        print(f"Saved combined plot: {combined_output}")


if __name__ == "__main__":
    main()
