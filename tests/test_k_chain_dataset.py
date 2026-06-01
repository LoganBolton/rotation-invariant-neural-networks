from __future__ import annotations

import pytest
import torch

from benchmarks.k_chain.generate_data.kchains import (
    as_hippynn_arrays,
    create_kchains,
    pair_distance_matrix,
)


def verify_k_chain(k: int, dist_hard_max: float) -> tuple[dict[str, torch.Tensor], list[int]]:
    graphs = create_kchains(k)
    arrays = as_hippynn_arrays(graphs)

    cutoff_pair_counts = []

    assert len(graphs) == 2
    assert arrays["Z"].shape == (2, k + 2)
    assert arrays["R"].shape == (2, k + 2, 3)
    assert arrays["T"].shape == (2, 1)
    assert arrays["central_atom_mask"].shape == (2, k + 2)

    expected_central_mask = torch.nn.functional.one_hot(
        torch.zeros(2, dtype=torch.long),
        k + 2,
    ).to(arrays["central_atom_mask"].dtype)

    assert torch.equal(arrays["central_atom_mask"], expected_central_mask)
    assert torch.equal(arrays["Z"], torch.ones_like(arrays["Z"]))
    assert torch.equal(arrays["T"].squeeze(-1).long(), torch.tensor([0, 1]))
    assert torch.allclose(arrays["R"].mean(dim=1), torch.zeros(2, 3), atol=1e-6)

    for graph in graphs:
        expected_edges = 2 * (k + 1)

        assert graph.k == k
        assert graph.edge_index.shape == (2, expected_edges)

        dmat = pair_distance_matrix(graph.R)
        cutoff_pairs = (dmat <= dist_hard_max) & ~torch.eye(k + 2, dtype=torch.bool)

        assert cutoff_pairs.any()
        cutoff_pair_counts.append(int(cutoff_pairs.sum().item()))

    return arrays, cutoff_pair_counts


@pytest.mark.parametrize("k", [2, 3, 4, 5, 8])
def test_k_chain_dataset(k: int) -> None:
    arrays, cutoff_pair_counts = verify_k_chain(k, dist_hard_max=6.5)

    assert arrays["Z"].shape[0] == 2
    assert len(cutoff_pair_counts) == 2
    assert all(count > 0 for count in cutoff_pair_counts)