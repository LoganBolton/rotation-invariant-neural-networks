"""Generate HTML viewers for benchmark datasets."""

from __future__ import annotations

import argparse
import sys
from math import pi
from pathlib import Path

import torch

BENCHMARKS_ROOT = Path(__file__).resolve().parent
if str(BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_ROOT))

from incompleteness.generate_data.incompleteness import (
    COUNTEREXAMPLE_NAMES,
    IncompletenessEnvironment,
    create_all_incompleteness_pairs,
    create_incompleteness_pair,
)
from k_chain.generate_data.kchains import create_kchains
from rotating_ring.generate_data.rotating_ring_dataset import create_rotating_ring_dataset
from rotating_ring.generate_data.rotating_ring_viewer import VIEWER_VERSION, write_ring_graph_viewer

DEFAULT_OUTPUT_DIR = BENCHMARKS_ROOT / "visualizations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["k_chain", "incompleteness", "rotating_ring"], required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--html",
        type=Path,
        default=None,
        help="Optional exact output path. For incompleteness this requires one --counterexample.",
    )

    parser.add_argument("--k", type=int, default=4, help="k-chain middle-node count.")
    parser.add_argument(
        "--counterexample",
        choices=("all", *COUNTEREXAMPLE_NAMES),
        default="all",
        help="Incompleteness counterexample to visualize.",
    )

    parser.add_argument("--ring-n-graphs", type=int, default=100)
    parser.add_argument("--ring-seed", type=int, default=0)
    parser.add_argument("--ring-n-inner", type=int, default=8)
    parser.add_argument("--ring-n-outer", type=int, default=8)
    parser.add_argument("--ring-viewer-max-graphs", type=int, default=100)
    parser.add_argument("--ring-outer-3d-rotation-deg", type=float, default=0.0)
    parser.add_argument("--ring-outer-3d-axis-deg", type=float, default=0.0)
    parser.add_argument("--ring-random-parameters", action="store_true")
    parser.add_argument("--ring-shuffle", action="store_true")
    parser.add_argument("--ring-add-inner-ring-edges", action="store_true")
    parser.add_argument("--ring-add-outer-ring-edges", action="store_true")
    return parser.parse_args()


def centered_positions(positions: torch.Tensor) -> torch.Tensor:
    return positions - positions.mean(dim=0, keepdim=True)


def graph_axis_range(position_sets: list[torch.Tensor]) -> list[float]:
    positions = torch.cat([centered_positions(positions) for positions in position_sets])
    extent = float(positions.abs().max().item())
    extent = max(1.0, extent * 1.1)
    return [-extent, extent]


def add_graph_traces(
    fig,
    *,
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    label: int,
    scene: str,
    title: str,
    node_colors: list[str] | None = None,
) -> None:
    import plotly.graph_objects as go

    positions = centered_positions(positions)
    scene_col = 1 if scene == "scene" else 2
    directed_edges = edge_index.t().tolist()
    undirected_edges = sorted({tuple(sorted((int(src), int(dst)))) for src, dst in directed_edges if src != dst})

    for src, dst in undirected_edges:
        distance = torch.dist(positions[src], positions[dst]).item()
        fig.add_trace(
            go.Scatter3d(
                x=[positions[src, 0].item(), positions[dst, 0].item()],
                y=[positions[src, 1].item(), positions[dst, 1].item()],
                z=[positions[src, 2].item(), positions[dst, 2].item()],
                mode="lines",
                line={"color": "#555555", "width": 6},
                hoverinfo="text",
                hovertext=f"edge {src}-{dst}<br>distance={distance:.5f}",
                showlegend=False,
            ),
            row=1,
            col=scene_col,
        )

    colors = node_colors or ["#0072B2"] * positions.shape[0]
    hovertext = [
        (
            f"{title}<br>class {label}<br>node {index}<br>"
            f"x={positions[index, 0].item():.6g}<br>"
            f"y={positions[index, 1].item():.6g}<br>"
            f"z={positions[index, 2].item():.6g}"
        )
        for index in range(positions.shape[0])
    ]
    fig.add_trace(
        go.Scatter3d(
            x=positions[:, 0].tolist(),
            y=positions[:, 1].tolist(),
            z=positions[:, 2].tolist(),
            mode="markers+text",
            marker={"color": colors, "size": 17, "line": {"color": "#222222", "width": 2}},
            text=[str(index) for index in range(positions.shape[0])],
            textfont={"color": "#111111", "size": 16},
            textposition="top center",
            hoverinfo="text",
            hovertext=hovertext,
            showlegend=False,
        ),
        row=1,
        col=scene_col,
    )


def write_two_graph_viewer(
    *,
    title: str,
    left_title: str,
    right_title: str,
    left_positions: torch.Tensor,
    right_positions: torch.Tensor,
    left_edges: torch.Tensor,
    right_edges: torch.Tensor,
    left_label: int,
    right_label: int,
    output_file: Path,
    left_colors: list[str] | None = None,
    right_colors: list[str] | None = None,
) -> None:
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        horizontal_spacing=0.03,
        subplot_titles=(left_title, right_title),
    )
    add_graph_traces(
        fig,
        positions=left_positions,
        edge_index=left_edges,
        label=left_label,
        scene="scene",
        title=left_title,
        node_colors=left_colors,
    )
    add_graph_traces(
        fig,
        positions=right_positions,
        edge_index=right_edges,
        label=right_label,
        scene="scene2",
        title=right_title,
        node_colors=right_colors,
    )

    shared_axis = {
        "showbackground": True,
        "backgroundcolor": "rgb(248,248,248)",
        "gridcolor": "rgb(218,218,218)",
        "zerolinecolor": "rgb(180,180,180)",
        "range": graph_axis_range([left_positions, right_positions]),
        "dtick": 1,
    }
    scene = {"xaxis": shared_axis, "yaxis": shared_axis, "zaxis": shared_axis, "aspectmode": "cube"}
    fig.update_layout(
        scene={**scene, "domain": {"x": [0.0, 0.485], "y": [0.0, 1.0]}},
        scene2={**scene, "domain": {"x": [0.515, 1.0], "y": [0.0, 1.0]}},
        margin={"l": 0, "r": 0, "t": 52, "b": 0},
        title={"text": title},
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_file, include_plotlyjs=True, full_html=True)


def visualize_k_chain(args: argparse.Namespace) -> list[Path]:
    graphs = create_kchains(args.k)
    output_file = args.html or args.output_dir / "k_chain" / f"k{int(args.k)}.html"
    write_two_graph_viewer(
        title=f"k-chain k={args.k}",
        left_title="class 0",
        right_title="class 1",
        left_positions=graphs[0].R,
        right_positions=graphs[1].R,
        left_edges=graphs[0].edge_index,
        right_edges=graphs[1].edge_index,
        left_label=graphs[0].label,
        right_label=graphs[1].label,
        output_file=output_file,
    )
    return [output_file]


def incompleteness_colors(environment: IncompletenessEnvironment) -> list[str]:
    colors = ["#D55E00"] + ["#0072B2"] * (environment.R.shape[0] - 1)
    colors[-1] = "#CC79A7"
    return colors


def visualize_incompleteness(args: argparse.Namespace) -> list[Path]:
    if args.counterexample == "all":
        if args.html is not None:
            raise ValueError("--html can only be used with one incompleteness --counterexample.")
        pairs_by_name = create_all_incompleteness_pairs()
    else:
        pairs_by_name = {args.counterexample: create_incompleteness_pair(args.counterexample)}

    output_files = []
    for name, environments in pairs_by_name.items():
        output_file = args.html or args.output_dir / "incompleteness" / f"{name}_hippynn.html"
        write_two_graph_viewer(
            title=f"{name} incompleteness",
            left_title=f"{name}: class 0",
            right_title=f"{name}: class 1",
            left_positions=environments[0].R,
            right_positions=environments[1].R,
            left_edges=environments[0].edge_index,
            right_edges=environments[1].edge_index,
            left_label=environments[0].label,
            right_label=environments[1].label,
            output_file=output_file,
            left_colors=incompleteness_colors(environments[0]),
            right_colors=incompleteness_colors(environments[1]),
        )
        output_files.append(output_file)
    return output_files


def visualize_rotating_ring(args: argparse.Namespace) -> list[Path]:
    output_file = (
        args.html
        or args.output_dir
        / "rotating_ring"
        / f"ring_{args.ring_n_inner}inner_{args.ring_n_outer}outer_{args.ring_n_graphs}.html"
    )
    environments = create_rotating_ring_dataset(
        n_graphs=args.ring_n_graphs,
        seed=args.ring_seed,
        n_inner=args.ring_n_inner,
        n_outer=args.ring_n_outer,
        outer_3d_rotation_range=(0.0, args.ring_outer_3d_rotation_deg * pi / 180.0),
        outer_3d_axis_angle=args.ring_outer_3d_axis_deg * pi / 180.0,
        smooth_order=not args.ring_random_parameters,
        shuffle=args.ring_shuffle,
        add_inner_ring_edges=args.ring_add_inner_ring_edges,
        add_outer_ring_edges=args.ring_add_outer_ring_edges,
    )
    write_ring_graph_viewer(environments, output_file, max_graphs=args.ring_viewer_max_graphs)
    return [output_file]


def main() -> None:
    args = parse_args()
    if args.dataset == "k_chain":
        output_files = visualize_k_chain(args)
    elif args.dataset == "incompleteness":
        output_files = visualize_incompleteness(args)
    elif args.dataset == "rotating_ring":
        output_files = visualize_rotating_ring(args)
    else:
        raise ValueError(f"Unknown dataset {args.dataset!r}.")

    for output_file in output_files:
        print(f"wrote {output_file}")
    if args.dataset == "rotating_ring":
        print(f"rotating-ring viewer version: {VIEWER_VERSION}")


if __name__ == "__main__":
    main()
