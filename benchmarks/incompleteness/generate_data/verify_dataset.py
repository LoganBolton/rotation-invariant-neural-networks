"""Verify and plot local-neighborhood incompleteness counterexamples."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from generate_data.incompleteness import (
    COUNTEREXAMPLE_NAMES,
    IncompletenessEnvironment,
    as_hippynn_arrays,
    body_order_signature,
    create_incompleteness_pair,
    pair_distance_matrix,
    star_edge_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counterexample",
        choices=("all", *COUNTEREXAMPLE_NAMES),
        default="all",
        help="Which counterexample pair to verify.",
    )
    parser.add_argument("--dist-hard-max", type=float, default=6.5, help="HIP-NN hard cutoff to inspect.")
    parser.add_argument("--plot", action="store_true", help="Save 3D visual plots of the environments.")
    parser.add_argument(
        "--plot-format",
        choices=("html", "png"),
        default="html",
        help="Save an interactive Plotly HTML plot by default, or a static matplotlib PNG.",
    )
    parser.add_argument(
        "--plot-source",
        choices=("hippynn", "raw"),
        default="hippynn",
        help="Plot centered HIP-NN R tensors by default, or raw environment.R coordinates.",
    )
    parser.add_argument(
        "--plot-cutoff-pairs",
        action="store_true",
        help="Deprecated: cutoff-pair overlays are no longer drawn in plots.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory for plot images. Defaults to this script directory.",
    )
    return parser.parse_args()


def selected_names(counterexample: str) -> tuple[str, ...]:
    if counterexample == "all":
        return COUNTEREXAMPLE_NAMES
    return (counterexample,)


def verify_pair(name: str, dist_hard_max: float) -> dict[str, object]:
    environments = create_incompleteness_pair(name)
    arrays = as_hippynn_arrays(environments)
    body_order = environments[0].indistinguishable_body_order
    n_nodes = environments[0].Z.shape[0]
    cutoff_pair_counts = []

    assert len(environments) == 2
    assert arrays["Z"].shape == (2, n_nodes)
    assert arrays["R"].shape == (2, n_nodes, 3)
    assert arrays["T"].shape == (2, 1)
    assert torch.equal(arrays["Z"], torch.ones_like(arrays["Z"]))
    assert torch.equal(arrays["T"].squeeze(-1).long(), torch.tensor([0, 1]))
    assert torch.allclose(arrays["R"].mean(dim=1), torch.zeros(2, 3), atol=1e-6)

    expected_edges = star_edge_index(n_nodes)
    for environment in environments:
        assert environment.name == name
        assert environment.indistinguishable_body_order == body_order
        assert torch.allclose(environment.R[0], torch.zeros(3), atol=1e-6)
        assert torch.equal(environment.edge_index, expected_edges)

        dmat = pair_distance_matrix(environment.R)
        cutoff_pairs = (dmat <= dist_hard_max) & ~torch.eye(n_nodes, dtype=torch.bool)
        assert cutoff_pairs.any()
        cutoff_pair_counts.append(int(cutoff_pairs.sum().item()))

    signature_0 = body_order_signature(environments[0])
    signature_1 = body_order_signature(environments[1])
    assert signature_0 == signature_1

    next_order_matches = None
    if body_order + 1 <= n_nodes:
        next_order_matches = body_order_signature(environments[0], body_order + 1) == body_order_signature(environments[1], body_order + 1)

    return {
        "arrays": arrays,
        "body_order": body_order,
        "n_nodes": n_nodes,
        "signature_count": len(signature_0),
        "cutoff_pair_counts": cutoff_pair_counts,
        "next_order_matches": next_order_matches,
    }


def _set_equal_3d_limits(axes, limit: float) -> None:
    from matplotlib.ticker import MultipleLocator

    for axis in axes:
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_zlim(-limit, limit)
        axis.set_box_aspect((1, 1, 1))
        axis.xaxis.set_major_locator(MultipleLocator(1))
        axis.yaxis.set_major_locator(MultipleLocator(1))
        axis.zaxis.set_major_locator(MultipleLocator(1))


def _shared_axis_limit(positions_by_sample: torch.Tensor) -> float:
    return max(float(positions_by_sample.abs().max().item()) * 1.1, 1.0)


def _standard_axis_limit(plot_source: str) -> float:
    all_positions = []
    for name in COUNTEREXAMPLE_NAMES:
        environments = create_incompleteness_pair(name)
        all_positions.append(_plot_positions(environments, plot_source))
    return _shared_axis_limit(torch.cat(all_positions, dim=1))


def _center_neighbor_pairs(positions: torch.Tensor) -> list[tuple[int, int]]:
    return [(0, node_index) for node_index in range(1, positions.shape[0])]


def _changed_node_indices(environments: list[IncompletenessEnvironment]) -> set[int]:
    if len(environments) != 2 or environments[0].R.shape != environments[1].R.shape:
        return set()

    changed = ~torch.isclose(environments[0].R, environments[1].R).all(dim=1)
    return {int(index) for index in torch.nonzero(changed, as_tuple=False).flatten().tolist()}


def _unique_cutoff_pairs(positions: torch.Tensor, dist_hard_max: float) -> list[tuple[int, int]]:
    dmat = pair_distance_matrix(positions)
    pairs = []
    for start in range(positions.shape[0]):
        for end in range(start + 1, positions.shape[0]):
            if dmat[start, end] <= dist_hard_max:
                pairs.append((start, end))
    return pairs


def _plot_positions(environments: list[IncompletenessEnvironment], plot_source: str) -> torch.Tensor:
    arrays = as_hippynn_arrays(environments, center=plot_source == "hippynn")
    return arrays["R"]


def plot_pair_png(
    environments: list[IncompletenessEnvironment],
    output_file: Path,
    *,
    plot_source: str,
    plot_cutoff_pairs: bool,
    dist_hard_max: float,
) -> None:
    """Save a static side-by-side 3D plot of the actual counterexample coordinates."""

    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib-cache")))

    import matplotlib.pyplot as plt

    positions_by_sample = _plot_positions(environments, plot_source)
    axis_limit = _standard_axis_limit(plot_source)

    fig = plt.figure(figsize=(10, 4.8), constrained_layout=True)
    axes = [fig.add_subplot(1, 2, index + 1, projection="3d") for index in range(2)]
    changed_nodes = _changed_node_indices(environments)

    for sample_index, (axis, environment) in enumerate(zip(axes, environments)):
        positions = positions_by_sample[sample_index]

        for start, end in _center_neighbor_pairs(positions):
            axis.plot(
                [positions[start, 0].item(), positions[end, 0].item()],
                [positions[start, 1].item(), positions[end, 1].item()],
                [positions[start, 2].item(), positions[end, 2].item()],
                color="#D55E00",
                linewidth=2.1,
                alpha=0.82,
                zorder=1,
            )

        axis.scatter(
            positions[0, 0],
            positions[0, 1],
            positions[0, 2],
            s=500,
            color="#D55E00",
            edgecolors="#222222",
            linewidths=1.0,
            label="node 0",
            depthshade=False,
            zorder=3,
        )
        unchanged_neighbor_indices = [index for index in range(1, positions.shape[0]) if index not in changed_nodes]
        changed_neighbor_indices = [index for index in range(1, positions.shape[0]) if index in changed_nodes]
        if unchanged_neighbor_indices:
            unchanged_positions = positions[unchanged_neighbor_indices]
            axis.scatter(
                unchanged_positions[:, 0],
                unchanged_positions[:, 1],
                unchanged_positions[:, 2],
                s=320,
                color="#0072B2",
                edgecolors="#222222",
                linewidths=0.9,
                label="node",
                depthshade=False,
                zorder=2,
            )
        if changed_neighbor_indices:
            changed_positions = positions[changed_neighbor_indices]
            axis.scatter(
                changed_positions[:, 0],
                changed_positions[:, 1],
                changed_positions[:, 2],
                s=440,
                color="#CC79A7",
                edgecolors="#222222",
                linewidths=1.2,
                label="changed node",
                depthshade=False,
                zorder=4,
            )

        for node_index, position in enumerate(positions):
            axis.text(
                position[0].item(),
                position[1].item(),
                position[2].item(),
                f" {node_index}",
                fontsize=9,
                weight="bold",
            )

        source_label = "HIP-NN R" if plot_source == "hippynn" else "raw R"
        axis.set_title(f"{environment.name}: class {environment.label} ({source_label})")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        axis.legend(loc="upper left", fontsize=8)
        axis.view_init(elev=20, azim=-55)

    _set_equal_3d_limits(axes, axis_limit)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def plot_pair_html(
    environments: list[IncompletenessEnvironment],
    output_file: Path,
    *,
    plot_source: str,
    plot_cutoff_pairs: bool,
    dist_hard_max: float,
) -> None:
    """Save an interactive side-by-side 3D plot of the actual counterexample coordinates."""

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise RuntimeError("Interactive plots require plotly. Install dependencies with `uv sync`.") from exc

    positions_by_sample = _plot_positions(environments, plot_source)
    source_label = "HIP-NN R" if plot_source == "hippynn" else "raw R"
    axis_limit = _standard_axis_limit(plot_source)
    changed_nodes = _changed_node_indices(environments)
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[f"{environment.name}: class {environment.label} ({source_label})" for environment in environments],
        horizontal_spacing=0.03,
    )

    for sample_index, environment in enumerate(environments):
        col = sample_index + 1
        positions = positions_by_sample[sample_index]
        node_indices = list(range(positions.shape[0]))
        colors = [
            "#D55E00" if node_index == 0 else "#CC79A7" if node_index in changed_nodes else "#0072B2"
            for node_index in node_indices
        ]
        sizes = [22 if node_index == 0 else 21 if node_index in changed_nodes else 17 for node_index in node_indices]
        hover_text = [
            (
                f"{environment.name} class {environment.label}<br>"
                f"node {node_index}<br>"
                f"x={position[0].item():.6g}<br>"
                f"y={position[1].item():.6g}<br>"
                f"z={position[2].item():.6g}"
            )
            for node_index, position in enumerate(positions)
        ]

        for start, end in _center_neighbor_pairs(positions):
            distance = torch.dist(positions[start], positions[end]).item()
            fig.add_trace(
                go.Scatter3d(
                    x=[positions[start, 0].item(), positions[end, 0].item()],
                    y=[positions[start, 1].item(), positions[end, 1].item()],
                    z=[positions[start, 2].item(), positions[end, 2].item()],
                    mode="lines",
                    line={"color": "#D55E00", "width": 7},
                    hovertext=f"center line 0-{end}<br>distance={distance:.6g}",
                    hoverinfo="text",
                    showlegend=False,
                ),
                row=1,
                col=col,
            )

        fig.add_trace(
            go.Scatter3d(
                x=positions[:, 0].tolist(),
                y=positions[:, 1].tolist(),
                z=positions[:, 2].tolist(),
                mode="markers+text",
                marker={"color": colors, "size": sizes, "line": {"color": "#222222", "width": 2}},
                text=[str(node_index) for node_index in node_indices],
                textposition="top center",
                textfont={"size": 18, "color": "#111111"},
                customdata=node_indices,
                hovertext=hover_text,
                hoverinfo="text",
                name=f"class {environment.label}",
                showlegend=False,
            ),
            row=1,
            col=col,
        )

    axis_style = {
        "showbackground": True,
        "backgroundcolor": "rgb(248,248,248)",
        "gridcolor": "rgb(218,218,218)",
        "zerolinecolor": "rgb(180,180,180)",
        "range": [-axis_limit, axis_limit],
        "dtick": 1,
    }
    fig.update_layout(
        title=f"{environments[0].name} coordinates",
        margin={"l": 0, "r": 0, "t": 52, "b": 0},
        scene={"xaxis": axis_style, "yaxis": axis_style, "zaxis": axis_style, "aspectmode": "cube"},
        scene2={"xaxis": axis_style, "yaxis": axis_style, "zaxis": axis_style, "aspectmode": "cube"},
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_file, include_plotlyjs=True, full_html=True)


def main() -> None:
    args = parse_args()
    plot_dir = args.plot_dir or Path(__file__).parent

    for name in selected_names(args.counterexample):
        result = verify_pair(name, args.dist_hard_max)
        arrays = result["arrays"]

        print(f"Verified {name} counterexample")
        print(f"  body order: {result['body_order']}")
        print(f"  Z shape: {tuple(arrays['Z'].shape)}, unique species: {torch.unique(arrays['Z']).tolist()}")
        print(f"  R shape: {tuple(arrays['R'].shape)}, per-graph centers: {arrays['R'].mean(dim=1).tolist()}")
        print(f"  T shape: {tuple(arrays['T'].shape)}, labels: {arrays['T'].squeeze(-1).tolist()}")
        print(f"  matching body-order fingerprint entries: {result['signature_count']}")
        if result["next_order_matches"] is not None:
            print(f"  next body-order fingerprint also matches: {result['next_order_matches']}")
        print(f"  directed local pairs within {args.dist_hard_max}: {result['cutoff_pair_counts']}")

        if args.plot:
            environments = create_incompleteness_pair(name)
            cutoff_suffix = "_cutoff" if args.plot_cutoff_pairs else ""
            plot_file = plot_dir / f"{name}_{args.plot_source}{cutoff_suffix}.{args.plot_format}"
            plot_fn = plot_pair_html if args.plot_format == "html" else plot_pair_png
            plot_fn(
                environments,
                plot_file,
                plot_source=args.plot_source,
                plot_cutoff_pairs=args.plot_cutoff_pairs,
                dist_hard_max=args.dist_hard_max,
            )
            print(f"  saved plot: {plot_file}")


if __name__ == "__main__":
    main()
