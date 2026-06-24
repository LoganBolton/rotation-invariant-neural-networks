"""Plot rotating-ring margin-accuracy comparisons across models.

Each input file is treated as one model. The plot uses graph type on the x axis
and l_max on the y axis. For two models, each cell is split diagonally so the
two outcomes can be compared directly.

uv run python benchmarks/rotating_ring/multi_model_comparison.py benchmarks/rotating_ring/results/equi_vs_hiphop_const_radius/equiformer.json benchmarks/rotating_ring/results/equi_vs_hiphop_const_radius/hiphop.json --labels Equiformer HIP-HOP --output benchmarks/rotating_ring/results/equi_vs_hiphop_const_radius/multi_model_comparison.png
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon, Rectangle


SUCCESS_COLOR = "#2ca25f"
FAILURE_COLOR = "#de2d26"
MISSING_COLOR = "#d9d9d9"
EDGE_COLOR = "#333333"
CELL_WIDTH = 1.45


@dataclass(frozen=True)
class RunResult:
    model: str
    graph_key: tuple[int, int] | str
    graph_label: str
    l_max: int
    margin_accuracy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="JSON or JSONL result files; one file per model.")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional model labels, in the same order as inputs. Defaults to file stems.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/rotating_ring/results/equi_vs_hiphop_const_radius/multi_model_comparison.png"),
        help="Path for the output PNG.",
    )
    parser.add_argument(
        "--aggregate",
        choices=("max", "min", "mean", "last"),
        default="max",
        help="How to combine duplicate runs for the same model/graph/l_max.",
    )
    parser.add_argument(
        "--title",
        default="Model margin-accuracy comparison",
        help="Plot title.",
    )
    parser.add_argument(
        "--min-l-max",
        type=int,
        default=2,
        help="Smallest l_max column to show. Defaults to 2, which drops l_max=0 and l_max=1.",
    )
    return parser.parse_args()


def read_json_or_jsonl(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL line") from exc
        return records


def result_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if isinstance(data, dict):
        for key in ("results", "runs"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [data]

    return []


def nested_get(data: dict[str, Any], paths: tuple[tuple[str, ...], ...]) -> Any:
    for path in paths:
        current: Any = data
        for part in path:
            if not isinstance(current, dict) or part not in current:
                break
            current = current[part]
        else:
            return current
    return None


def infer_l_max(record: dict[str, Any]) -> int | None:
    value = nested_get(
        record,
        (
            ("config", "model", "lmax"),
            ("config", "lmax"),
            ("config", "l_max"),
            ("model", "lmax"),
            ("model", "l_max"),
            ("lmax",),
            ("l_max",),
        ),
    )
    if value is not None:
        return int(value)

    name = str(record.get("experiment_name", ""))
    match = re.search(r"(?:^|[_-])l(?:max)?_?(\d+)(?:$|[_-])", name)
    if match:
        return int(match.group(1))
    return None


def infer_margin_accuracy(record: dict[str, Any]) -> float | None:
    value = nested_get(
        record,
        (
            ("metrics", "margin_accuracy"),
            ("metrics", "mean_margin_accuracy"),
            ("summary", "margin_accuracy"),
            ("summary", "best_margin_accuracy"),
            ("margin_accuracy",),
            ("mean_margin_accuracy",),
            ("best_margin_accuracy",),
        ),
    )
    return None if value is None else float(value)


def infer_graph(record: dict[str, Any]) -> tuple[tuple[int, int] | str, str] | None:
    inner = nested_get(
        record,
        (
            ("config", "graph", "ring_n_inner"),
            ("config", "dataset", "ring_n_inner"),
            ("config", "ring_n_inner"),
            ("ring_n_inner",),
        ),
    )
    outer = nested_get(
        record,
        (
            ("config", "graph", "ring_n_outer"),
            ("config", "dataset", "ring_n_outer"),
            ("config", "ring_n_outer"),
            ("ring_n_outer",),
        ),
    )
    if inner is not None and outer is not None:
        inner_i = int(inner)
        outer_i = int(outer)
        return (inner_i, outer_i), f"{inner_i} inner / {outer_i} outer"

    name = str(record.get("experiment_name", ""))
    match = re.search(r"inner[_-]?(\d+).*outer[_-]?(\d+)", name)
    if match:
        inner_i = int(match.group(1))
        outer_i = int(match.group(2))
        return (inner_i, outer_i), f"{inner_i} inner / {outer_i} outer"
    return None


def collect_results(path: Path, model: str) -> list[RunResult]:
    results: list[RunResult] = []
    skipped = 0
    for record in result_records(read_json_or_jsonl(path)):
        graph = infer_graph(record)
        l_max = infer_l_max(record)
        margin_accuracy = infer_margin_accuracy(record)
        if graph is None or l_max is None or margin_accuracy is None:
            skipped += 1
            continue
        graph_key, graph_label = graph
        results.append(
            RunResult(
                model=model,
                graph_key=graph_key,
                graph_label=graph_label,
                l_max=l_max,
                margin_accuracy=margin_accuracy,
            )
        )

    if skipped:
        print(f"Skipped {skipped} records from {path} that were missing graph, l_max, or margin accuracy.")
    return results


def aggregate_values(values: list[float], method: str) -> float:
    if method == "max":
        return max(values)
    if method == "min":
        return min(values)
    if method == "mean":
        return mean(values)
    if method == "last":
        return values[-1]
    raise ValueError(f"Unknown aggregate method: {method}")


def sort_graph_key(key: tuple[int, int] | str) -> tuple[int, int, str]:
    if isinstance(key, tuple):
        return key[0], key[1], ""
    return 10**9, 10**9, str(key)


def result_color(value: float | None) -> str:
    if value is None:
        return MISSING_COLOR
    return SUCCESS_COLOR if value == 1.0 else FAILURE_COLOR


def add_model_marker(ax: plt.Axes, x: float, y: float, marker: str, model_index: int) -> None:
    ax.text(
        x,
        y,
        marker,
        ha="center",
        va="center",
        color="#111111",
        fontsize=8.5,
        fontweight="bold",
    )


def model_markers(models: list[str]) -> list[str]:
    markers: list[str] = []
    used: set[str] = set()
    for index, model in enumerate(models):
        marker = next((char.upper() for char in model if char.isalnum()), "")
        if not marker or marker in used:
            marker = chr(ord("A") + index)
        if marker in used:
            marker = str(index + 1)
        markers.append(marker)
        used.add(marker)
    return markers


def draw_two_model_cell(
    ax: plt.Axes,
    x: float,
    y: int,
    values: list[float | None],
    markers: list[str],
) -> None:
    half_width = CELL_WIDTH / 2.0
    ax.add_patch(Rectangle((x - half_width, y - 0.5), CELL_WIDTH, 1.0, facecolor="none", edgecolor=EDGE_COLOR, linewidth=0.9))
    ax.add_patch(
        Polygon(
            [(x - half_width, y - 0.5), (x + half_width, y - 0.5), (x - half_width, y + 0.5)],
            closed=True,
            facecolor=result_color(values[0]),
            edgecolor="none",
        )
    )
    ax.add_patch(
        Polygon(
            [(x + half_width, y + 0.5), (x + half_width, y - 0.5), (x - half_width, y + 0.5)],
            closed=True,
            facecolor=result_color(values[1]),
            edgecolor="none",
        )
    )
    ax.plot([x - half_width, x + half_width], [y + 0.5, y - 0.5], color=EDGE_COLOR, linewidth=0.9)
    add_model_marker(ax, x - 0.22 * CELL_WIDTH, y - 0.17, markers[0], 0)
    add_model_marker(ax, x + 0.22 * CELL_WIDTH, y + 0.17, markers[1], 1)


def draw_multi_model_cell(
    ax: plt.Axes,
    x: float,
    y: int,
    values: list[float | None],
    markers: list[str],
) -> None:
    half_width = CELL_WIDTH / 2.0
    width = CELL_WIDTH / len(values)
    ax.add_patch(Rectangle((x - half_width, y - 0.5), CELL_WIDTH, 1.0, facecolor="none", edgecolor=EDGE_COLOR, linewidth=0.9))
    for index, value in enumerate(values):
        ax.add_patch(
            Rectangle(
                (x - half_width + index * width, y - 0.5),
                width,
                1.0,
                facecolor=result_color(value),
                edgecolor=EDGE_COLOR,
                linewidth=0.35,
            )
        )
        add_model_marker(ax, x - half_width + (index + 0.5) * width, y, markers[index], index)


def inner_group_boundaries(graphs: list[tuple[int, int] | str]) -> list[float]:
    boundaries: list[float] = []
    for index, (previous, current) in enumerate(zip(graphs, graphs[1:]), start=0):
        if not isinstance(previous, tuple) or not isinstance(current, tuple):
            continue
        if previous[0] != current[0]:
            boundaries.append(index + 0.5)
    return boundaries


def plot_results(
    results: list[RunResult],
    models: list[str],
    output: Path,
    aggregate: str,
    title: str,
    min_l_max: int,
) -> None:
    grouped: dict[tuple[str, tuple[int, int] | str, int], list[float]] = {}
    graph_labels: dict[tuple[int, int] | str, str] = {}
    for result in results:
        grouped.setdefault((result.model, result.graph_key, result.l_max), []).append(result.margin_accuracy)
        graph_labels[result.graph_key] = result.graph_label

    graphs = sorted(graph_labels, key=sort_graph_key)
    l_max_values = sorted({result.l_max for result in results if result.l_max >= min_l_max}, reverse=True)
    if not graphs or not l_max_values:
        raise ValueError("No plottable results found.")

    matrix: dict[tuple[str, tuple[int, int] | str, int], float] = {
        key: aggregate_values(values, aggregate) for key, values in grouped.items()
    }

    x_positions = [index * CELL_WIDTH for index in range(len(graphs))]

    fig_width = max(8.0, 1.08 * len(graphs) + 3.8)
    fig_height = max(5.0, 0.55 * len(l_max_values) + 2.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    markers = model_markers(models)

    for y_index, l_max in enumerate(l_max_values):
        for x_position, graph in zip(x_positions, graphs, strict=True):
            values = [matrix.get((model, graph, l_max)) for model in models]
            if len(models) == 2:
                draw_two_model_cell(ax, x_position, y_index, values, markers)
            else:
                draw_multi_model_cell(ax, x_position, y_index, values, markers)

    for boundary in inner_group_boundaries(graphs):
        boundary_x = boundary * CELL_WIDTH
        ax.plot(
            [boundary_x, boundary_x],
            [-0.5, len(l_max_values) - 0.5],
            color="#111111",
            linewidth=2.8,
            solid_capstyle="butt",
        )

    ax.set_title(title)
    ax.set_xlabel("graph type")
    ax.set_ylabel("l_max")
    ax.set_xticks(x_positions, [graph_labels[graph] for graph in graphs], rotation=45, ha="right")
    ax.set_yticks(range(len(l_max_values)), [str(value) for value in l_max_values])
    ax.set_xlim(x_positions[0] - CELL_WIDTH / 2.0, x_positions[-1] + CELL_WIDTH / 2.0)
    ax.set_ylim(len(l_max_values) - 0.5, -0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", length=0)

    legend_items = [
        Patch(facecolor=SUCCESS_COLOR, edgecolor=EDGE_COLOR, label="margin_accuracy = 1.0"),
        Patch(facecolor=FAILURE_COLOR, edgecolor=EDGE_COLOR, label="margin_accuracy = 0.5"),
        Patch(facecolor=MISSING_COLOR, edgecolor=EDGE_COLOR, label="missing run"),
    ]
    if len(models) == 2:
        legend_items.extend(
            [
                Patch(facecolor="none", edgecolor="none", label=f"{markers[0]}: {models[0]} (upper-left)"),
                Patch(facecolor="none", edgecolor="none", label=f"{markers[1]}: {models[1]} (lower-right)"),
            ]
        )
    else:
        legend_items.extend(
            Patch(facecolor="none", edgecolor="none", label=f"{marker}: {model}")
            for marker, model in zip(markers, models, strict=True)
        )
    ax.legend(handles=legend_items, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.inputs):
        raise SystemExit("--labels must have the same number of entries as inputs.")

    models = args.labels if args.labels is not None else [path.stem for path in args.inputs]
    results: list[RunResult] = []
    for path, model in zip(args.inputs, models, strict=True):
        model_results = collect_results(path, model)
        print(f"Read {len(model_results)} plottable records for {model} from {path}.")
        results.extend(model_results)

    plot_results(results, models, args.output, args.aggregate, args.title, args.min_l_max)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
