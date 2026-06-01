from __future__ import annotations

import pytest
import torch

from benchmarks.incompleteness.generate_data.incompleteness import (
    COUNTEREXAMPLE_NAMES,
    as_hippynn_arrays,
    body_order_signature,
    create_incompleteness_pair,
    pair_distance_matrix,
    star_edge_index,
)


def verify_pair(name: str, dist_hard_max: float) -> dict[str, object]:
    environments = create_incompleteness_pair(name)
    arrays = as_hippynn_arrays(environments)

    body_order = environments[0].indistinguishable_body_order
    n_nodes = environments[0].Z.shape[0]

    assert len(environments) == 2
    assert arrays["Z"].shape == (2, n_nodes)
    assert arrays["R"].shape == (2, n_nodes, 3)
    assert arrays["T"].shape == (2, 1)
    assert arrays["central_atom_mask"].shape == (2, n_nodes)

    expected_central_mask = torch.nn.functional.one_hot(
        torch.zeros(2, dtype=torch.long),
        n_nodes,
    ).to(arrays["central_atom_mask"].dtype)

    assert torch.equal(arrays["central_atom_mask"], expected_central_mask)
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