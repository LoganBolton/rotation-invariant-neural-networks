"""Verify and plot local-neighborhood incompleteness counterexamples."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from incompleteness import (
    COUNTEREXAMPLE_NAMES,
    IncompletenessEnvironment,
    body_order_signature,
    create_all_incompleteness_pairs,
    create_incompleteness_pair,
    pair_distance_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counterexample", choices=COUNTEREXAMPLE_NAMES, default=None)
    parser.add_argument("--coordinate-set", choices=("v2", "original"), default="v2")
    parser.add_argument(
        "--dist-hard-max",
        type=float,
        default=None,
        help="Optional hard cutoff to check against max center-leaf and min leaf-leaf distances.",
    )
    parser.add_argument("--plot", action="store_true", help="Write Plotly HTML visualizations.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Plot output directory. Defaults to <coordinate-set>_visualizations next to this script.",
    )
    return parser.parse_args()


def cutoff_gap(environments: list[IncompletenessEnvironment]) -> tuple[float, float]:
    max_center_leaf = 0.0
    min_leaf_leaf = float("inf")
    for environment in environments:
        distances = pair_distance_matrix(environment.R)
        center_leaf = distances[0, 1:]
        leaf_leaf = distances[1:, 1:]
        max_center_leaf = max(max_center_leaf, float(center_leaf.max().item()))
        if leaf_leaf.numel() > 1:
            mask = ~torch.eye(leaf_leaf.shape[0], dtype=torch.bool)
            min_leaf_leaf = min(min_leaf_leaf, float(leaf_leaf[mask].min().item()))
    return max_center_leaf, min_leaf_leaf


def edge_distance_rows(environment: IncompletenessEnvironment) -> list[tuple[str, str, float]]:
    distances = pair_distance_matrix(environment.R)
    rows = []
    for leaf_index in range(1, environment.R.shape[0]):
        rows.append(("center-leaf", f"0-{leaf_index}", float(distances[0, leaf_index].item())))
    for first_leaf in range(1, environment.R.shape[0]):
        for second_leaf in range(first_leaf + 1, environment.R.shape[0]):
            rows.append(("leaf-leaf", f"{first_leaf}-{second_leaf}", float(distances[first_leaf, second_leaf].item())))
    return rows


def print_distance_table(name: str, environments: list[IncompletenessEnvironment]) -> None:
    print(f"\n{name} adjacency distances")
    print("| class | edge type | edge | distance |")
    print("| --- | --- | ---: | ---: |")
    for environment in environments:
        for edge_type, edge, distance in edge_distance_rows(environment):
            print(f"| {environment.label} | {edge_type} | {edge} | {distance:.6f} |")


def verify_pair(
    name: str,
    environments: list[IncompletenessEnvironment],
    *,
    dist_hard_max: float | None = None,
) -> None:
    if len(environments) != 2:
        raise ValueError(f"{name} should contain exactly 2 environments, got {len(environments)}.")

    order = environments[0].indistinguishable_body_order
    signature0 = body_order_signature(environments[0], order)
    signature1 = body_order_signature(environments[1], order)
    if signature0 != signature1:
        raise ValueError(f"{name} differs at its declared indistinguishable body order {order}.")

    max_center_leaf, min_leaf_leaf = cutoff_gap(environments)
    gap = min_leaf_leaf - max_center_leaf
    print_distance_table(name, environments)
    print(
        f"{name:22s} order={order} signature=ok "
        f"max_center_leaf={max_center_leaf:.4f} min_leaf_leaf={min_leaf_leaf:.4f} gap={gap:.4f}"
    )
    if dist_hard_max is None:
        print(f"valid dist_hard_max window: {max_center_leaf:.6f} < dist_hard_max < {min_leaf_leaf:.6f}")
    else:
        valid = max_center_leaf < dist_hard_max < min_leaf_leaf
        status = "ok" if valid else "invalid"
        print(
            f"dist_hard_max={dist_hard_max:.6f}: {status} "
            f"({max_center_leaf:.6f} < {dist_hard_max:.6f} < {min_leaf_leaf:.6f})"
        )


def centered_positions(environment: IncompletenessEnvironment) -> torch.Tensor:
    return environment.R - environment.R.mean(dim=0, keepdim=True)


def add_environment_traces(fig, environment: IncompletenessEnvironment, scene: str) -> None:
    import plotly.graph_objects as go

    positions = centered_positions(environment)
    colors = ["#D55E00"] + ["#0072B2"] * (positions.shape[0] - 1)
    colors[-1] = "#CC79A7"
    sizes = [22] + [17] * (positions.shape[0] - 1)
    sizes[-1] = 21

    for neighbor_index in range(1, positions.shape[0]):
        distance = torch.dist(positions[0], positions[neighbor_index]).item()
        fig.add_trace(
            go.Scatter3d(
                x=[positions[0, 0].item(), positions[neighbor_index, 0].item()],
                y=[positions[0, 1].item(), positions[neighbor_index, 1].item()],
                z=[positions[0, 2].item(), positions[neighbor_index, 2].item()],
                mode="lines",
                line={"color": "#D55E00", "width": 7},
                hoverinfo="text",
                hovertext=f"center line 0-{neighbor_index}<br>distance={distance:.5f}",
                showlegend=False,
            ),
            row=1,
            col=1 if scene == "scene" else 2,
        )

    hovertext = [
        (
            f"{environment.name} class {environment.label}<br>"
            f"node {index}<br>"
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
            marker={"color": colors, "size": sizes, "line": {"color": "#222222", "width": 2}},
            text=[str(index) for index in range(positions.shape[0])],
            textfont={"color": "#111111", "size": 18},
            textposition="top center",
            hoverinfo="text",
            hovertext=hovertext,
            customdata=list(range(positions.shape[0])),
            name=f"class {environment.label}",
            showlegend=False,
        ),
        row=1,
        col=1 if scene == "scene" else 2,
    )


def axis_range(environments: list[IncompletenessEnvironment]) -> list[float]:
    positions = torch.cat([centered_positions(environment) for environment in environments])
    extent = float(positions.abs().max().item())
    extent = max(1.0, extent * 1.1)
    return [-extent, extent]


def plot_pair(name: str, environments: list[IncompletenessEnvironment], output_file: Path) -> None:
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        horizontal_spacing=0.03,
        subplot_titles=(
            f"{name}: class 0 (HIP-NN R)",
            f"{name}: class 1 (HIP-NN R)",
        ),
    )
    add_environment_traces(fig, environments[0], "scene")
    add_environment_traces(fig, environments[1], "scene2")

    shared_axis = {
        "showbackground": True,
        "backgroundcolor": "rgb(248,248,248)",
        "gridcolor": "rgb(218,218,218)",
        "zerolinecolor": "rgb(180,180,180)",
        "range": axis_range(environments),
        "dtick": 1,
    }
    scene = {"xaxis": shared_axis, "yaxis": shared_axis, "zaxis": shared_axis, "aspectmode": "cube"}
    fig.update_layout(
        scene={**scene, "domain": {"x": [0.0, 0.485], "y": [0.0, 1.0]}},
        scene2={**scene, "domain": {"x": [0.515, 1.0], "y": [0.0, 1.0]}},
        margin={"l": 0, "r": 0, "t": 52, "b": 0},
        title={"text": f"{name} coordinates"},
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_file, include_plotlyjs=True, full_html=True)


def main() -> None:
    args = parse_args()
    if args.counterexample is None:
        pairs_by_name = create_all_incompleteness_pairs(coordinate_set=args.coordinate_set)
    else:
        pairs_by_name = {
            args.counterexample: create_incompleteness_pair(args.counterexample, coordinate_set=args.coordinate_set)
        }

    for name, environments in pairs_by_name.items():
        verify_pair(name, environments, dist_hard_max=args.dist_hard_max)

    if args.plot:
        output_dir = args.output_dir or Path(__file__).with_name(f"{args.coordinate_set}_visualizations")
        for name, environments in pairs_by_name.items():
            output_file = output_dir / f"{name}_hippynn.html"
            plot_pair(name, environments, output_file)
            print(f"wrote {output_file}")


if __name__ == "__main__":
    main()
