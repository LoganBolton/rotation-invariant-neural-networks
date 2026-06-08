from __future__ import annotations

import pytest
import torch

from benchmarks.incompleteness.generate_data.incompleteness import (
    COUNTEREXAMPLE_NAMES,
    as_hippynn_arrays,
    as_padded_hippynn_arrays,
    body_order_signature,
    create_all_incompleteness_pairs,
    create_incompleteness_pair,
    pair_distance_matrix,
    star_edge_index,
)


def assert_center_leaf_edges(arrays: dict[str, torch.Tensor]) -> None:
    species = arrays["Z"]
    edge_index = arrays["edge_index"]
    edge_indices = arrays["edge_indices"]

    assert edge_index.dtype == torch.long
    assert edge_index.ndim == 2
    assert edge_index.shape[0] == 2
    assert edge_indices.dtype == torch.long
    assert edge_indices.ndim == 3
    assert edge_indices.shape[:2] == (species.shape[0], 2)

    real_atoms = species != 0
    atom_system_indices, atom_local_indices = torch.nonzero(real_atoms, as_tuple=True)
    edge_first, edge_second = edge_index

    assert edge_index.shape[1] > 0
    assert edge_first.min() >= 0
    assert edge_second.min() >= 0
    assert edge_first.max() < real_atoms.sum()
    assert edge_second.max() < real_atoms.sum()

    first_system = atom_system_indices[edge_first]
    second_system = atom_system_indices[edge_second]
    first_local = atom_local_indices[edge_first]
    second_local = atom_local_indices[edge_second]
    central_local = torch.zeros(species.shape[0], dtype=torch.long)

    assert torch.equal(first_system, second_system)
    first_is_center = first_local == central_local[first_system]
    second_is_center = second_local == central_local[second_system]
    assert torch.all(first_is_center | second_is_center)
    assert not torch.any((~first_is_center) & (~second_is_center))

    expected_counts = 2 * (real_atoms.sum(dim=1) - 1)
    actual_counts = torch.bincount(first_system, minlength=species.shape[0])
    assert torch.equal(actual_counts, expected_counts)

    rebuilt_edges = []
    atom_offset = 0
    for sample_index, n_real in enumerate(real_atoms.sum(dim=1).tolist()):
        sample_edges = edge_indices[sample_index]
        present = (sample_edges[0] >= 0) & (sample_edges[1] >= 0)
        assert torch.all(sample_edges[:, ~present] == -1)
        assert present.sum() == expected_counts[sample_index]
        assert sample_edges[:, present].max() < n_real
        rebuilt_edges.append(sample_edges[:, present] + atom_offset)
        atom_offset += n_real

    assert torch.equal(torch.cat(rebuilt_edges, dim=1), edge_index)


def verify_pair(name: str, dist_hard_max: float) -> dict[str, object]:
    environments = create_incompleteness_pair(name)
    arrays = as_hippynn_arrays(environments)

    body_order = environments[0].indistinguishable_body_order
    n_nodes = environments[0].Z.shape[0]

    assert len(environments) == 2
    assert arrays["Z"].shape == (2, n_nodes)
    assert arrays["R"].shape == (2, n_nodes, 3)
    assert arrays["T"].shape == (2, 1)
    assert_center_leaf_edges(arrays)
    assert torch.equal(arrays["Z"], torch.ones_like(arrays["Z"]))
    assert torch.equal(arrays["T"].squeeze(-1).long(), torch.tensor([0, 1]))
    assert torch.allclose(arrays["R"].mean(dim=1), torch.zeros(2, 3), atol=1e-6)

    expected_edges = star_edge_index(n_nodes)
    cutoff_pair_counts = []

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
        next_order_matches = body_order_signature(
            environments[0],
            body_order + 1,
        ) == body_order_signature(
            environments[1],
            body_order + 1,
        )

    return {
        "arrays": arrays,
        "body_order": body_order,
        "n_nodes": n_nodes,
        "signature_count": len(signature_0),
        "cutoff_pair_counts": cutoff_pair_counts,
        "next_order_matches": next_order_matches,
    }


@pytest.mark.parametrize("name", COUNTEREXAMPLE_NAMES)
def test_incompleteness_counterexample_dataset(name: str) -> None:
    result = verify_pair(name, dist_hard_max=6.5)

    assert result["body_order"] >= 2
    assert result["n_nodes"] >= result["body_order"]
    assert result["signature_count"] > 0
    assert len(result["cutoff_pair_counts"]) == 2


def test_padded_incompleteness_edges_do_not_touch_padding() -> None:
    pairs_by_name = create_all_incompleteness_pairs()
    environments = [environment for name in COUNTEREXAMPLE_NAMES for environment in pairs_by_name[name]]
    arrays = as_padded_hippynn_arrays(environments)

    assert arrays["Z"].shape[1] > min(environment.Z.shape[0] for environment in environments)
    assert_center_leaf_edges(arrays)


def test_two_body_edges_exclude_leaf_leaf_edges_at_large_cutoff() -> None:
    environments = create_incompleteness_pair("two_body")
    arrays = as_hippynn_arrays(environments)
    _atom_system_indices, atom_local_indices = torch.nonzero(arrays["Z"] != 0, as_tuple=True)

    first_local = atom_local_indices[arrays["edge_index"][0]]
    second_local = atom_local_indices[arrays["edge_index"][1]]
    assert not torch.any((first_local != 0) & (second_local != 0))

    for environment in environments:
        dmat = pair_distance_matrix(environment.R)
        dense_large_cutoff_pairs = (dmat <= 100.0) & ~torch.eye(environment.R.shape[0], dtype=torch.bool)
        assert torch.any(dense_large_cutoff_pairs[1:, 1:])
