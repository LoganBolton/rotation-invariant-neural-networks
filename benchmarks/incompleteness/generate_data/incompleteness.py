"""Dataset helpers for local-neighborhood incompleteness counterexamples."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import cos, pi, sin

import torch


COUNTEREXAMPLE_NAMES = ("two_body", "three_body", "four_body_nonchiral", "four_body_chiral")


@dataclass(frozen=True)
class IncompletenessEnvironment:
    """A single local neighborhood with node 0 as the central atom."""

    name: str
    indistinguishable_body_order: int
    label: int
    Z: torch.Tensor
    R: torch.Tensor
    edge_index: torch.Tensor


def star_edge_index(n_nodes: int) -> torch.Tensor:
    """Return an undirected star edge index centered at node 0."""

    neighbors = torch.arange(1, n_nodes, dtype=torch.long)
    center = torch.zeros_like(neighbors)
    return torch.cat(
        [
            torch.stack([center, neighbors]),
            torch.stack([neighbors, center]),
        ],
        dim=1,
    )


def _environment(name: str, body_order: int, label: int, positions: list[list[float]]) -> IncompletenessEnvironment:
    pos = torch.tensor(positions, dtype=torch.get_default_dtype())
    n_nodes = pos.shape[0]
    return IncompletenessEnvironment(
        name=name,
        indistinguishable_body_order=body_order,
        label=label,
        Z=torch.ones(n_nodes, dtype=torch.long),
        R=pos,
        edge_index=star_edge_index(n_nodes),
    )


def _rotation_y_matrix(angle: float) -> torch.Tensor:
    """Match e3nn.o3.matrix_y for the fixed notebook rotation without depending on e3nn."""

    c = cos(angle)
    s = sin(angle)
    return torch.tensor(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=torch.get_default_dtype(),
    )


def _rotate_row(vector: tuple[float, float, float], matrix: torch.Tensor) -> list[float]:
    return (torch.tensor(vector, dtype=matrix.dtype) @ matrix).tolist()


def create_two_body_envs() -> list[IncompletenessEnvironment]:
    """Pair indistinguishable by center-neighbor 2-body scalars."""

    return [
        _environment(
            "two_body",
            2,
            0,
            [
                [0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [3.0, 0.0, 4.0],
            ],
        ),
        _environment(
            "two_body",
            2,
            1,
            [
                [0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [-5.0, 0.0, 0.0],
            ],
        ),
    ]


def create_three_body_envs() -> list[IncompletenessEnvironment]:
    """Pair indistinguishable by centered 3-body distance/angle scalars."""

    a_x, a_y, a_z = 5.0, 0.0, 5.0
    b_x, b_y, b_z = 5.0, 5.0, 5.0
    c_x, c_y, c_z = 0.0, 5.0, 5.0

    return [
        _environment(
            "three_body",
            3,
            0,
            [
                [0.0, 0.0, 0.0],
                [a_x, a_y, a_z],
                [b_x, b_y, b_z],
                [-b_x, -b_y, b_z],
                [c_x, c_y, c_z],
            ],
        ),
        _environment(
            "three_body",
            3,
            1,
            [
                [0.0, 0.0, 0.0],
                [a_x, a_y, a_z],
                [b_x, b_y, b_z],
                [-b_x, -b_y, b_z],
                [c_x, -c_y, c_z],
            ],
        ),
    ]


def create_four_body_nonchiral_envs() -> list[IncompletenessEnvironment]:
    """Pair indistinguishable by non-oriented centered 4-body scalars."""

    q = _rotation_y_matrix(pi / 10)
    a1 = [3.0, 2.0, -4.0]
    a2 = [0.0, 2.0, 5.0]
    a3 = [0.0, 2.0, -5.0]
    b1 = _rotate_row((3.0, -2.0, -4.0), q)
    b2 = _rotate_row((0.0, -2.0, 5.0), q)
    b3 = _rotate_row((0.0, -2.0, -5.0), q)

    common = [
        [0.0, 0.0, 0.0],
        a1,
        a2,
        a3,
        b1,
        b2,
        b3,
    ]
    return [
        _environment("four_body_nonchiral", 4, 0, common + [[0.0, 5.0, 0.0]]),
        _environment("four_body_nonchiral", 4, 1, common + [[0.0, -5.0, 0.0]]),
    ]


def create_four_body_chiral_envs() -> list[IncompletenessEnvironment]:
    """Pair from the notebook's chiral 4-body counterexample."""

    common = [
        [0.0, 0.0, 0.0],
        [3.0, 0.0, -4.0],
        [0.0, 0.0, 5.0],
        [0.0, 0.0, -5.0],
    ]
    return [
        _environment("four_body_chiral", 4, 0, common + [[0.0, 5.0, 0.0]]),
        _environment("four_body_chiral", 4, 1, common + [[0.0, -5.0, 0.0]]),
    ]


def create_incompleteness_pair(name: str) -> list[IncompletenessEnvironment]:
    """Create one named counterexample pair."""

    builders = {
        "two_body": create_two_body_envs,
        "three_body": create_three_body_envs,
        "four_body_nonchiral": create_four_body_nonchiral_envs,
        "four_body_chiral": create_four_body_chiral_envs,
    }
    try:
        return builders[name]()
    except KeyError as exc:
        valid = ", ".join(COUNTEREXAMPLE_NAMES)
        raise ValueError(f"Unknown counterexample {name!r}. Expected one of: {valid}.") from exc


def create_all_incompleteness_pairs() -> dict[str, list[IncompletenessEnvironment]]:
    """Create every incompleteness counterexample pair."""

    return {name: create_incompleteness_pair(name) for name in COUNTEREXAMPLE_NAMES}


def as_hippynn_arrays(
    environments: list[IncompletenessEnvironment],
    *,
    center: bool = True,
) -> dict[str, torch.Tensor]:
    """Stack a same-size environment list into arrays compatible with HIP-NN keys."""

    if not environments:
        raise ValueError("Cannot stack an empty environment list.")

    n_nodes = environments[0].Z.shape[0]
    if any(env.Z.shape != (n_nodes,) or env.R.shape != (n_nodes, 3) for env in environments):
        raise ValueError("All environments must have the same number of nodes to stack without padding.")

    positions = torch.stack([env.R for env in environments])
    if center:
        positions = positions - positions.mean(dim=1, keepdim=True)

    return {
        "Z": torch.stack([env.Z for env in environments]),
        "R": positions,
        "T": torch.tensor([[env.label] for env in environments], dtype=torch.get_default_dtype()),
    }


def as_padded_hippynn_arrays(
    environments: list[IncompletenessEnvironment],
    *,
    center: bool = True,
) -> dict[str, torch.Tensor]:
    """Stack variable-size environments into padded arrays compatible with HIP-NN keys."""

    if not environments:
        raise ValueError("Cannot stack an empty environment list.")

    max_nodes = max(env.Z.shape[0] for env in environments)
    species = torch.zeros((len(environments), max_nodes), dtype=torch.long)
    positions = torch.zeros((len(environments), max_nodes, 3), dtype=torch.get_default_dtype())

    for sample_index, environment in enumerate(environments):
        n_nodes = environment.Z.shape[0]
        species[sample_index, :n_nodes] = environment.Z
        sample_positions = environment.R
        if center:
            sample_positions = sample_positions - sample_positions.mean(dim=0, keepdim=True)
        positions[sample_index, :n_nodes] = sample_positions

    return {
        "Z": species,
        "R": positions,
        "T": torch.tensor([[env.label] for env in environments], dtype=torch.get_default_dtype()),
    }


def pair_distance_matrix(positions: torch.Tensor) -> torch.Tensor:
    """Dense pairwise distances for a single local environment."""

    return torch.linalg.vector_norm(positions[:, None, :] - positions[None, :, :], dim=-1)


def body_order_signature(
    environment: IncompletenessEnvironment,
    body_order: int | None = None,
    *,
    decimals: int = 4,
) -> list[tuple[float, ...]]:
    """Return the unordered centered body-order distance fingerprint.

    Body order 2 uses one neighbor at a time, body order 3 uses neighbor pairs,
    and body order 4 uses neighbor triples. Each entry is the sorted set of
    pairwise distances among the central node and that neighbor subset.
    """

    order = environment.indistinguishable_body_order if body_order is None else body_order
    if order < 2:
        raise ValueError(f"body_order must be at least 2, got {order}.")

    subset_size = order - 1
    neighbor_indices = range(1, environment.R.shape[0])
    signatures = []
    for combo in combinations(neighbor_indices, subset_size):
        indices = (0, *combo)
        positions = environment.R[list(indices)]
        distances = []
        for i, j in combinations(range(len(indices)), 2):
            distances.append(round(float(torch.dist(positions[i], positions[j]).item()), decimals))
        signatures.append(tuple(sorted(distances)))
    return sorted(signatures)
