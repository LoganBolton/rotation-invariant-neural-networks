"""Dataset helpers for the k-chain distinguishability toy problem."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class KChainGraph:
    """A single canonical k-chain geometry."""

    k: int
    label: int
    Z: torch.Tensor
    R: torch.Tensor
    edge_index: torch.Tensor


def chain_edge_index(n_nodes: int) -> torch.Tensor:
    """Return an undirected path edge index for diagnostic checks."""

    forward = torch.arange(n_nodes - 1, dtype=torch.long)
    backward = forward + 1
    return torch.cat(
        [
            torch.stack([forward, backward]),
            torch.stack([backward, forward]),
        ],
        dim=1,
    )


def _centered_positions(k: int, same_side: bool) -> torch.Tensor:
    left_endpoint_x = 4.0 if same_side else -4.0
    pos = torch.tensor(
        [[left_endpoint_x, -3.0, 0.0]]
        + [[0.0, 5.0 * i, 0.0] for i in range(k)]
        + [[4.0, 5.0 * (k - 1) + 3.0, 0.0]],
        dtype=torch.get_default_dtype(),
    )
    return pos - pos.mean(dim=0, keepdim=True)


def create_kchains(k: int) -> list[KChainGraph]:
    """Create the two canonical k-chain graphs for a fixed k.

    Label 0 has endpoints on opposite sides of the vertical chain.
    Label 1 has endpoints on the same side of the vertical chain.
    """

    if k < 2:
        raise ValueError(f"k must be at least 2, got {k}.")

    n_nodes = k + 2
    species = torch.ones(n_nodes, dtype=torch.long)
    edges = chain_edge_index(n_nodes)

    return [
        KChainGraph(k=k, label=0, Z=species.clone(), R=_centered_positions(k, same_side=False), edge_index=edges),
        KChainGraph(k=k, label=1, Z=species.clone(), R=_centered_positions(k, same_side=True), edge_index=edges),
    ]


def as_hippynn_arrays(graphs: list[KChainGraph]) -> dict[str, torch.Tensor]:
    """Stack k-chain graphs into arrays compatible with hippynn database keys."""

    if not graphs:
        raise ValueError("Cannot stack an empty graph list.")

    n_nodes = graphs[0].Z.shape[0]
    if any(g.Z.shape != (n_nodes,) or g.R.shape != (n_nodes, 3) for g in graphs):
        raise ValueError("All graphs must have the same number of nodes to stack without padding.")

    species = torch.stack([g.Z for g in graphs])

    return {
        "Z": species,
        "R": torch.stack([g.R for g in graphs]),
        "T": torch.tensor([[g.label] for g in graphs], dtype=torch.get_default_dtype()),
        "edge_indices": torch.stack([g.edge_index for g in graphs]),
    }


def pair_distance_matrix(positions: torch.Tensor) -> torch.Tensor:
    """Dense pairwise distances for a single geometry."""

    return torch.linalg.vector_norm(positions[:, None, :] - positions[None, :, :], dim=-1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the k-chain benchmark pair.")
    parser.add_argument("--k", type=int, default=4, help="Number of middle chain nodes.")
    parser.add_argument(
        "--dist-hard-max",
        type=float,
        default=6.5,
        help="Cutoff used only for reporting dense neighbor counts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for torch.save() of the HIP-NN tensor dictionary.",
    )
    return parser.parse_args()


def _cutoff_pair_count(positions: torch.Tensor, dist_hard_max: float) -> int:
    distances = pair_distance_matrix(positions)
    mask = (distances <= dist_hard_max) & ~torch.eye(positions.shape[0], dtype=torch.bool)
    return int(mask.sum().item())


def main() -> None:
    args = _parse_args()
    graphs = create_kchains(args.k)
    arrays = as_hippynn_arrays(graphs)

    print(f"k-chain dataset: k={args.k}, graphs={len(graphs)}, nodes={arrays['Z'].shape[1]}")
    print("| label | path edges | dense cutoff pairs | diameter |")
    print("| ---: | ---: | ---: | ---: |")
    for graph in graphs:
        distances = pair_distance_matrix(graph.R)
        print(
            f"| {graph.label} | {graph.edge_index.shape[1]} | "
            f"{_cutoff_pair_count(graph.R, args.dist_hard_max)} | {float(distances.max().item()):.4f} |"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(arrays, args.output)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
