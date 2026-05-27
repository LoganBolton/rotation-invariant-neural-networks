"""Verify the basic k-chain dataset construction."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from kchains import KChainGraph, as_hippynn_arrays, create_kchains, pair_distance_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=4, help="Number of middle chain nodes.")
    parser.add_argument("--dist-hard-max", type=float, default=6.5, help="HIP-NN hard cutoff to inspect.")
    parser.add_argument("--plot", action="store_true", help="Save a visual plot of the two geometries.")
    parser.add_argument(
        "--plot-file",
        type=Path,
        default=None,
        help="Path for the plot image. Defaults to kchain_k<K>.png next to this script.",
    )
    return parser.parse_args()


def verify(k: int, dist_hard_max: float) -> tuple[dict[str, torch.Tensor], list[int]]:
    graphs = create_kchains(k)
    arrays = as_hippynn_arrays(graphs)
    cutoff_pair_counts = []

    assert len(graphs) == 2
    assert arrays["Z"].shape == (2, k + 2)
    assert arrays["R"].shape == (2, k + 2, 3)
    assert arrays["T"].shape == (2, 1)
    assert torch.equal(arrays["Z"], torch.ones_like(arrays["Z"]))
    assert torch.equal(arrays["T"].squeeze(-1).long(), torch.tensor([0, 1]))
    assert torch.allclose(arrays["R"].mean(dim=1), torch.zeros(2, 3), atol=1e-6)

    for graph in graphs:
        expected_edges = 2 * (k + 1)
        assert graph.edge_index.shape == (2, expected_edges)
        dmat = pair_distance_matrix(graph.R)
        cutoff_pairs = (dmat <= dist_hard_max) & ~torch.eye(k + 2, dtype=torch.bool)
        assert cutoff_pairs.any()
        cutoff_pair_counts.append(int(cutoff_pairs.sum().item()))

    return arrays, cutoff_pair_counts


def plot_graphs(graphs: list[KChainGraph], output_file: Path) -> None:
    """Save a side-by-side plot of the two k-chain geometries."""

    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib-cache")))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 5), constrained_layout=True, sharex=True, sharey=True)
    titles = {
        0: "Class 0: endpoints on opposite sides",
        1: "Class 1: endpoints on same side",
    }

    all_positions = torch.cat([graph.R for graph in graphs], dim=0)
    x_min, y_min = all_positions[:, :2].min(dim=0).values.tolist()
    x_max, y_max = all_positions[:, :2].max(dim=0).values.tolist()
    x_pad = max(1.0, 0.15 * (x_max - x_min))
    y_pad = max(1.0, 0.08 * (y_max - y_min))

    for axis, graph in zip(axes, graphs):
        positions = graph.R
        x = positions[:, 0].numpy()
        y = positions[:, 1].numpy()

        for start, end in graph.edge_index[:, : graph.k + 1].T.tolist():
            axis.plot([x[start], x[end]], [y[start], y[end]], color="0.70", linewidth=2, zorder=1)

        axis.scatter(x[1:-1], y[1:-1], s=80, color="#4C78A8", label="chain node", zorder=2)
        axis.scatter(x[[0, -1]], y[[0, -1]], s=130, color="#F58518", label="endpoint", zorder=3)

        for node_index, (node_x, node_y) in enumerate(zip(x, y)):
            axis.annotate(
                str(node_index),
                (node_x, node_y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                weight="bold",
            )

        axis.axvline(0, color="0.88", linewidth=1, zorder=0)
        axis.axhline(0, color="0.88", linewidth=1, zorder=0)
        axis.set_title(titles[graph.label])
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_xlim(x_min - x_pad, x_max + x_pad)
        axis.set_ylim(y_min - y_pad, y_max + y_pad)
        axis.legend(loc="upper left", fontsize=8)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    graphs = create_kchains(args.k)
    arrays, cutoff_pair_counts = verify(args.k, args.dist_hard_max)

    print(f"Verified k-chain dataset for k={args.k}")
    print(f"Z shape: {tuple(arrays['Z'].shape)}, unique species: {torch.unique(arrays['Z']).tolist()}")
    print(f"R shape: {tuple(arrays['R'].shape)}, per-graph centers: {arrays['R'].mean(dim=1).tolist()}")
    print(f"T shape: {tuple(arrays['T'].shape)}, labels: {arrays['T'].squeeze(-1).tolist()}")
    print(f"Directed local pairs within {args.dist_hard_max}: {cutoff_pair_counts}")
    print("First graph positions:")
    print(arrays["R"][0])
    print("Second graph positions:")
    print(arrays["R"][1])

    if args.plot:
        plot_file = args.plot_file
        if plot_file is None:
            plot_file = Path(__file__).with_name(f"kchain_k{args.k}.png")
        plot_graphs(graphs, plot_file)
        print(f"Saved plot: {plot_file}")


if __name__ == "__main__":
    main()
