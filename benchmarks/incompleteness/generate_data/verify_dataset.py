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


def _set_equal_3d_limits(axes, environments: list[IncompletenessEnvironment]) -> None:
    positions = torch.cat([environment.R for environment in environments], dim=0)
    mins = positions.min(dim=0).values
    maxs = positions.max(dim=0).values
    centers = (mins + maxs) / 2
    radius = float((maxs - mins).max().item() / 2)
    radius = max(radius, 1.0)

    for axis in axes:
        axis.set_xlim(float(centers[0] - radius), float(centers[0] + radius))
        axis.set_ylim(float(centers[1] - radius), float(centers[1] + radius))
        axis.set_zlim(float(centers[2] - radius), float(centers[2] + radius))


def plot_pair(environments: list[IncompletenessEnvironment], output_file: Path) -> None:
    """Save a side-by-side 3D plot of a counterexample pair."""

    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib-cache")))

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 4.8), constrained_layout=True)
    axes = [fig.add_subplot(1, 2, index + 1, projection="3d") for index in range(2)]

    for axis, environment in zip(axes, environments):
        positions = environment.R
        center = positions[0]

        for neighbor_index in range(1, positions.shape[0]):
            neighbor = positions[neighbor_index]
            axis.plot(
                [center[0].item(), neighbor[0].item()],
                [center[1].item(), neighbor[1].item()],
                [center[2].item(), neighbor[2].item()],
                color="0.72",
                linewidth=1.8,
                zorder=1,
            )

        axis.scatter(center[0], center[1], center[2], s=150, color="#222222", label="center", depthshade=False)
        axis.scatter(
            positions[1:, 0],
            positions[1:, 1],
            positions[1:, 2],
            s=90,
            color="#4C78A8",
            label="neighbor",
            depthshade=False,
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

        axis.set_title(f"{environment.name}: class {environment.label}")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        axis.legend(loc="upper left", fontsize=8)
        axis.view_init(elev=20, azim=-55)

    _set_equal_3d_limits(axes, environments)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


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
            plot_file = plot_dir / f"{name}.png"
            plot_pair(environments, plot_file)
            print(f"  saved plot: {plot_file}")


if __name__ == "__main__":
    main()
