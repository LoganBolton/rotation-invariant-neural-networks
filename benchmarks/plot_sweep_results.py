"""Plot sweep results saved from sweep.py JSON outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib-cache")))

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


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
    parser.add_argument("input_path", type=Path, help="Sweep result JSON, index.json, or directory containing JSON results.")
    parser.add_argument("--output", type=Path, default=None, help="Output image path for a single-result plot.")
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
    parser.add_argument(
        "--cutoffs",
        type=float,
        nargs="+",
        default=None,
        help="Only display these hard cutoff values.",
    )
    parser.add_argument(
        "--hide-max-diameter",
        action="store_true",
        help="Do not include max diameter annotations in counterexample titles.",
    )
    parser.add_argument("--no-title", action="store_true", help="Do not draw the plot title.")
    return parser.parse_args()


def parse_model_label(log_file: Path) -> str:
    return model_label(load_result_records(log_file)[0], log_file.stem)


def model_label(record: dict[str, Any], fallback: str) -> str:
    config = record.get("config", {})
    model = str(config.get("model", ""))
    l_max = config.get("l_max")
    n_max = config.get("n_max")
    if model == "hipnn" or (l_max == 0 and n_max == 1):
        return "HIP-NN"
    if model == "hipnnvec":
        return "HIP-NN-Vec"
    if l_max is not None and n_max is not None:
        return f"HIP-HOP-NN\nl={int(l_max)}, n={int(n_max)}"
    return fallback


def model_sort_key(log_file: Path) -> tuple[int, int, str]:
    config = load_result_records(log_file)[0].get("config", {})
    return (int(config.get("l_max", 10_000)), int(config.get("n_max", 10_000)), log_file.stem)


def result_json_files(result_dir: Path) -> list[Path]:
    files = sorted(
        (path for path in result_dir.glob("*.json") if path.name != "index.json"),
        key=model_sort_key,
    )
    if not files:
        files = sorted(
            (path for path in result_dir.glob("**/*.json") if path.name != "index.json"),
            key=model_sort_key,
        )
    if not files and (result_dir / "index.json").exists():
        files = [result_dir / "index.json"]
    if not files:
        raise ValueError(f"No sweep JSON results found in {result_dir}.")
    return files


def output_metric_name(metric: str, average_cutoffs: bool) -> str:
    return f"{metric}_average_cutoffs" if average_cutoffs else metric


def load_result_records(result_file: Path) -> list[dict[str, Any]]:
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    return payload.get("results", [payload])


def parse_results(result_file: Path) -> list[dict[str, float | str]]:
    return parse_result_records(load_result_records(result_file), result_file)


def parse_result_records(records: list[dict[str, Any]], source: Path) -> list[dict[str, float | str]]:
    rows_by_cell: dict[tuple[str, float, int], dict[str, list[float]]] = {}

    for record in records:
        for run in record.get("runs", []):
            if run.get("status") != "ok":
                continue

            model_config = run["config"]["model"]
            metrics = run["metrics"]
            item = dataset_item_label(run)
            cutoff = float(model_config["dist_hard_max"])
            layers = int(model_config["n_interaction_layers"])
            cell = rows_by_cell.setdefault(
                (item, cutoff, layers),
                {"accuracy": [], "margin_accuracy": []},
            )
            cell["accuracy"].append(float(metrics["accuracy"]))
            cell["margin_accuracy"].append(float(metrics["margin_accuracy"]))

    rows = [
        {
            "counterexample": item,
            "cutoff": cutoff,
            "layers": layers,
            "accuracy": float(np.mean(values["accuracy"])),
            "margin_accuracy": float(np.mean(values["margin_accuracy"])),
            "success_rate": float(np.mean([value >= 1.0 for value in values["margin_accuracy"]])),
        }
        for (item, cutoff, layers), values in rows_by_cell.items()
    ]

    if not rows:
        raise ValueError(f"No sweep result rows found in {source}.")

    return sorted(rows, key=lambda row: (str(row["counterexample"]), float(row["cutoff"]), int(row["layers"])))


def dataset_item_label(run: dict[str, Any]) -> str:
    dataset = run.get("dataset", run["config"]["dataset"])
    if "counterexample" in dataset:
        return str(dataset["counterexample"])
    if "k" in dataset:
        return str(dataset["k"])
    if dataset.get("dataset") == "rotating_ring":
        n_inner = dataset.get("ring_n_inner")
        n_outer = dataset.get("ring_n_outer")
        if n_inner is not None and n_outer is not None:
            return f"inner{n_inner}_outer{n_outer}"
    return str(dataset.get("dataset", "dataset"))


def filter_cutoffs(
    rows: list[dict[str, float | str]],
    cutoffs: list[float] | None,
) -> list[dict[str, float | str]]:
    if cutoffs is None:
        return rows

    cutoff_set = set(cutoffs)
    return [row for row in rows if float(row["cutoff"]) in cutoff_set]


def metric_colormap(metric: str) -> mcolors.Colormap:
    if metric in {"accuracy", "success_rate"}:
        return mcolors.ListedColormap(["#c62828", "#2e7d32"])
    return mcolors.LinearSegmentedColormap.from_list(
        "failure_to_success",
        ["#c62828", "#f7f7f7", "#2e7d32"],
    )


def metric_norm(metric: str) -> mcolors.Normalize:
    if metric in {"accuracy", "success_rate"}:
        return mcolors.BoundaryNorm([-0.5, 0.999999, 1.5], 2)
    return mcolors.Normalize(vmin=0.5, vmax=1.0)


def square_cells(average_cutoffs: bool, cutoffs: list[float | str]) -> bool:
    return average_cutoffs or len(cutoffs) == 1


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


def item_title(item: str, hide_max_diameter: bool) -> str:
    geometry = COUNTEREXAMPLE_GEOMETRY.get(item)
    title = item.replace("_", " ")
    if geometry is not None and not hide_max_diameter:
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
    hide_max_diameter: bool,
    no_title: bool,
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

    cmap = metric_colormap(metric)
    norm = metric_norm(metric)

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

        axis.imshow(
            grid,
            cmap=cmap,
            norm=norm,
            origin="lower",
            aspect="equal" if square_cells(average_cutoffs, cutoffs) else "auto",
        )

        axis.set_title(item_title(counterexample, hide_max_diameter))

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

    if not no_title:
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
    hide_max_diameter: bool,
    no_title: bool,
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
    use_square_cells = square_cells(average_cutoffs, cutoffs)
    fig_height = max((0.9 if use_square_cells else 2.15) * n_models + 1.0, 3.2)
    fig, axes = plt.subplots(
        n_models,
        n_items,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        gridspec_kw={"hspace": 0.04 if use_square_cells else 0.18},
        squeeze=False,
    )

    cmap = metric_colormap(metric)
    norm = metric_norm(metric)
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

            axis.imshow(
                grid,
                cmap=cmap,
                norm=norm,
                origin="lower",
                aspect="equal" if use_square_cells else "auto",
            )

            if y_model == 0:
                axis.set_title(item_title(item, hide_max_diameter))
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

    if not no_title:
        fig.suptitle(f"Model comparison: {metric.replace('_', ' ')}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.input_path.is_dir():
        result_files = result_json_files(args.input_path)
        combined_output = args.combined_output
        if combined_output is None:
            combined_output = args.input_path / f"combined_{output_metric_name(args.metric, args.average_cutoffs)}_grid.png"
        rows_by_model = [
            (parse_model_label(result_file), filter_cutoffs(parse_results(result_file), args.cutoffs))
            for result_file in result_files
        ]
        plot_combined_results(
            rows_by_model,
            args.metric,
            combined_output,
            args.min_layers,
            args.average_cutoffs,
            args.hide_max_diameter,
            args.no_title,
        )
        print(f"Saved combined plot: {combined_output}")
        return

    if args.input_path.name == "index.json":
        records = load_result_records(args.input_path)
        combined_output = args.combined_output or args.output
        if combined_output is None:
            combined_output = args.input_path.with_name(
                f"combined_{output_metric_name(args.metric, args.average_cutoffs)}_grid.png"
            )
        rows_by_model = [
            (
                model_label(record, str(record.get("experiment_name", args.input_path.stem))),
                filter_cutoffs(parse_result_records([record], args.input_path), args.cutoffs),
            )
            for record in records
        ]
        plot_combined_results(
            rows_by_model,
            args.metric,
            combined_output,
            args.min_layers,
            args.average_cutoffs,
            args.hide_max_diameter,
            args.no_title,
        )
        print(f"Saved combined plot: {combined_output}")
        return

    output_file = args.output
    if output_file is None:
        output_file = args.input_path.with_name(
            f"{args.input_path.stem}_{output_metric_name(args.metric, args.average_cutoffs)}_grid.png"
        )

    rows = filter_cutoffs(parse_results(args.input_path), args.cutoffs)
    model_label = parse_model_label(args.input_path)
    plot_results(
        rows,
        args.metric,
        output_file,
        args.min_layers,
        model_label,
        args.average_cutoffs,
        args.hide_max_diameter,
        args.no_title,
    )
    print(f"Saved plot: {output_file}")

    sibling_results = result_json_files(args.input_path.parent)
    if len(sibling_results) > 1:
        combined_output = args.combined_output
        if combined_output is None:
            combined_output = args.input_path.parent / (
                f"combined_{output_metric_name(args.metric, args.average_cutoffs)}_grid.png"
            )
        rows_by_model = [
            (parse_model_label(result_file), filter_cutoffs(parse_results(result_file), args.cutoffs))
            for result_file in sibling_results
        ]
        plot_combined_results(
            rows_by_model,
            args.metric,
            combined_output,
            args.min_layers,
            args.average_cutoffs,
            args.hide_max_diameter,
            args.no_title,
        )
        print(f"Saved combined plot: {combined_output}")


if __name__ == "__main__":
    main()
