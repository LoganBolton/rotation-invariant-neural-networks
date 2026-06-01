"""Focused checks for HIP-NN central-atom readout."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "hippynn-local"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

os.environ.setdefault("HIPPYNN_USE_CUSTOM_KERNELS", "False")

from benchmarks.incompleteness.generate_data.incompleteness import (  # noqa: E402
    atom_mask_from_local,
    as_padded_hippynn_arrays,
    create_all_incompleteness_pairs,
)
from hippynn.layers.targets import HEnergy  # noqa: E402
from hippynn.networks.hipnn import Hipnn  # noqa: E402


def test_masked_henergy_shapes_and_batched_indices() -> None:
    feature_0 = torch.arange(16, dtype=torch.get_default_dtype()).unsqueeze(1)
    feature_1 = 100 + torch.arange(16, dtype=torch.get_default_dtype()).unsqueeze(1)
    system_index = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2], dtype=torch.long)
    central_atom_mask = torch.zeros(16, 1)
    central_atom_mask[torch.tensor([2, 6, 12])] = 1.0

    readout = HEnergy((1, 1), n_target=1)
    with torch.no_grad():
        for layer in readout.layers:
            layer.weight.fill_(1.0)
            if layer.bias is not None:
                layer.bias.zero_()

    total_output, atom_outputs, partial_sums, total_hier, atom_hier, mol_hier, batch_hier = readout(
        [feature_0, feature_1], system_index, 3, central_atom_mask
    )

    assert total_output.shape == (3, 1)
    assert atom_outputs.shape == (16, 1)
    for partial_sum in partial_sums:
        assert partial_sum.shape == (3, 1)
    for diagnostic in (total_hier, atom_hier, mol_hier, batch_hier):
        assert not torch.isnan(diagnostic).any()

    expected = feature_0[torch.tensor([2, 6, 12])] + feature_1[torch.tensor([2, 6, 12])]
    assert torch.equal(total_output, expected)


def test_masked_henergy_does_not_sum_over_all_atoms() -> None:
    central_atom_mask = torch.zeros(5, 1)
    central_atom_mask[torch.tensor([1, 3])] = 1.0
    features = [torch.zeros(5, 1)]
    features[0][central_atom_mask.squeeze(1).bool()] = 1.0

    readout = HEnergy((1,), n_target=1)
    normal_readout = HEnergy((1,), n_target=1)
    with torch.no_grad():
        readout.layers[0].weight.fill_(1.0)
        normal_readout.layers[0].weight.fill_(1.0)

    system_index = torch.zeros(5, dtype=torch.long)
    central_output, *_ = readout(features, system_index, 1, central_atom_mask)
    changed_features = [features[0].clone()]
    changed_features[0][torch.tensor([0, 2, 4])] = 100.0
    changed_central_output, *_ = readout(changed_features, system_index, 1, central_atom_mask)

    assert torch.equal(central_output, changed_central_output)

    normal_output, *_ = normal_readout(changed_features, system_index, 1)
    assert normal_output.item() != central_output.item()


def test_padded_dataset_central_mask() -> None:
    pairs_by_name = create_all_incompleteness_pairs()
    environments = [environment for environments in pairs_by_name.values() for environment in environments]
    arrays = as_padded_hippynn_arrays(environments)

    expected = atom_mask_from_local(arrays["Z"], 0)
    assert torch.equal(arrays["central_atom_mask"], expected)
    assert arrays["central_atom_mask"].shape == arrays["Z"].shape
    assert torch.equal(arrays["central_atom_mask"].sum(dim=1), torch.ones(len(environments)))


def test_noncentral_features_receive_gradients_through_message_passing() -> None:
    network = Hipnn(
        n_features=4,
        n_sensitivities=2,
        dist_soft_min=0.5,
        dist_soft_max=1.5,
        dist_hard_max=2.0,
        n_atom_layers=0,
        n_interaction_layers=1,
        n_input_features=2,
        resnet=False,
    )
    readout = HEnergy(network.feature_sizes, n_target=1)

    with torch.no_grad():
        interaction = network.interaction_layers[0]
        interaction.int_weights.fill_(1.0)
        interaction.selfint.weight.zero_()
        interaction.selfint.bias.zero_()
        readout.layers[0].weight.zero_()
        readout.layers[1].weight.fill_(1.0)
        readout.layers[1].bias.zero_()

    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    pair_first = torch.tensor([0], dtype=torch.long)
    pair_second = torch.tensor([1], dtype=torch.long)
    pair_dist = torch.tensor([1.0])

    hierarchy_features = network(features, pair_first, pair_second, pair_dist)
    system_index = torch.zeros(2, dtype=torch.long)
    central_atom_mask = torch.tensor([[1.0], [0.0]])
    total_output, *_ = readout(hierarchy_features, system_index, 1, central_atom_mask)
    total_output.sum().backward()

    assert features.grad is not None
    assert features.grad[1].abs().sum() > 0


if __name__ == "__main__":
    test_masked_henergy_shapes_and_batched_indices()
    test_masked_henergy_does_not_sum_over_all_atoms()
    test_padded_dataset_central_mask()
    test_noncentral_features_receive_gradients_through_message_passing()
