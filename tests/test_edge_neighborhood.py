"""Focused checks for explicit edge neighborhoods."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "hippynn-local"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

os.environ.setdefault("HIPPYNN_USE_CUSTOM_KERNELS", "False")

from benchmarks.run_models.train import (  # noqa: E402
    EDGE_NEIGHBORHOOD_DIST_HARD_MAX,
    load_dataset,
    make_model,
    model_forward_args,
)
from hippynn.layers.targets import HEnergy  # noqa: E402
from hippynn.networks.hipnn import Hipnn  # noqa: E402


def edge_neighborhood_args(model: str) -> SimpleNamespace:
    return SimpleNamespace(
        dataset="incompleteness",
        k=4,
        counterexample="two_body",
        model=model,
        neighborhood_cutoff="edges",
        n_features=4,
        n_sensitivities=2,
        dist_soft_min=1.0,
        dist_soft_max=6.0,
        dist_hard_max=100.0,
        n_interaction_layers=2,
        n_atom_layers=0,
        l_max=1,
        n_max=2,
    )


def test_edge_neighborhood_uses_nonrestrictive_sensitivity_cutoff() -> None:
    edge_args = edge_neighborhood_args("hipnn")
    edge_args.dist_hard_max = 5.0
    edge_args.dist_soft_max = None
    model = make_model(edge_args)
    hipnn_module = next(module for module in model.moddict.values() if isinstance(module, Hipnn))
    sensitivity = hipnn_module.blocks[0][0].base_layer.sensitivity

    assert sensitivity.hard_max_dist == EDGE_NEIGHBORHOOD_DIST_HARD_MAX
    assert torch.equal(sensitivity.mu.detach(), torch.tensor([[6.0, 1.0]]))


def test_neighbor_features_receive_gradients_through_message_passing() -> None:
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
    total_output, *_ = readout(hierarchy_features, system_index, 1)
    total_output.sum().backward()

    assert features.grad is not None
    assert features.grad[1].abs().sum() > 0


def test_hipnn_edge_neighborhood_forward_shape() -> None:
    args = edge_neighborhood_args("hipnn")
    arrays, _description = load_dataset(args)
    model = make_model(args)

    with torch.no_grad():
        (logits,) = model(*model_forward_args(args, arrays))

    assert logits.shape == arrays["T"].shape


def test_hiphop_edge_neighborhood_forward_shape() -> None:
    args = edge_neighborhood_args("hiphop")
    arrays, _description = load_dataset(args)
    model = make_model(args)

    with torch.no_grad():
        (logits,) = model(*model_forward_args(args, arrays))

    assert logits.shape == arrays["T"].shape
