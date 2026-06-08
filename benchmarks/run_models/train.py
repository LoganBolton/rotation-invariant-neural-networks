"""Train HIP-NN on two-sample geometric benchmark tasks."""

from __future__ import annotations

import argparse
import os
import random
import sys
import warnings
from pathlib import Path

BENCHMARKS_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_ROOT))

import torch

from incompleteness.generate_data.incompleteness import (
    COUNTEREXAMPLE_NAMES,
    as_hippynn_arrays as as_incompleteness_arrays,
    as_padded_hippynn_arrays as as_padded_incompleteness_arrays,
    create_all_incompleteness_pairs,
    create_incompleteness_pair,
)
from k_chain.generate_data.kchains import as_hippynn_arrays as as_kchain_arrays
from k_chain.generate_data.kchains import create_kchains


os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib-cache")))
os.environ.setdefault("HIPPYNN_USE_CUSTOM_KERNELS", "False")

EDGE_NEIGHBORHOOD_DIST_HARD_MAX = 1.0e6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["k_chain", "incompleteness"], default="k_chain", help="Benchmark dataset to train on.")
    parser.add_argument("--k", type=int, default=4, help="Number of middle chain nodes.")
    parser.add_argument(
        "--counterexample",
        choices=("all", *COUNTEREXAMPLE_NAMES),
        default="all",
        help="Incompleteness counterexample to train on when --dataset incompleteness.",
    )
    parser.add_argument("--epochs", type=int, default=4000, help="Number of full-batch training epochs.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--model", choices=["hipnn", "hipnnvec", "hiphop"], default="hipnn", help="Network architecture to train.")
    parser.add_argument(
        "--neighborhood-cutoff",
        choices=["cutoff", "edges"],
        default="cutoff",
        help="Build interaction neighbors from HIP-NN cutoffs or dataset-defined edges.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--n-interaction-layers", type=int, default=3, help="HIP-NN interaction layers.")
    parser.add_argument("--n-atom-layers", type=int, default=2, help="Atom layers inside each interaction block.")
    parser.add_argument("--n-features", type=int, default=32, help="HIP-NN feature width.")
    parser.add_argument("--n-sensitivities", type=int, default=32, help="Number of sensitivity functions.")
    parser.add_argument("--dist-soft-min", type=float, default=1.0)
    parser.add_argument(
        "--dist-soft-max",
        type=float,
        default=None,
        help="Sensitivity soft maximum. Defaults to 6.0 for local cutoffs and 0.85 * dist-hard-max otherwise.",
    )
    parser.add_argument("--dist-hard-max", type=float, default=6.5)
    parser.add_argument("--l-max", type=int, default=2, help="HIP-HOP angular order.")
    parser.add_argument("--n-max", type=int, default=3, help="HIP-HOP radial tensor order.")
    parser.add_argument("--stop-at-accuracy", type=float, default=1.0, help="Early-stop once this margin accuracy is reached.")
    parser.add_argument("--success-margin", type=float, default=0.1, help="Report margin accuracy using this logit margin.")
    return parser.parse_args()


def edge_pair_tensors(arrays: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert compressed atom-indexed edges into HIP-NN pair tensors."""

    missing = {key for key in ("Z", "R", "edge_index") if key not in arrays}
    if missing:
        raise ValueError(f"Explicit neighbor topology requires dataset arrays: {sorted(missing)}.")

    species = arrays["Z"]
    positions = arrays["R"]
    edge_index = arrays["edge_index"]

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"Expected edge_index shape [2, n_edges], got {tuple(edge_index.shape)}.")

    real_atoms = species != 0
    real_flat_indices = torch.nonzero(real_atoms.reshape(-1), as_tuple=False).squeeze(1)
    n_real_atoms = real_flat_indices.numel()
    if edge_index.numel() > 0 and ((edge_index < 0).any() or (edge_index >= n_real_atoms).any()):
        raise ValueError("edge_index contains compressed atom indices outside the real atom range.")

    pair_first = edge_index[0].to(dtype=torch.long)
    pair_second = edge_index[1].to(dtype=torch.long)
    atom_positions = positions.reshape(-1, 3)[real_flat_indices]
    pair_coord = atom_positions[pair_first] - atom_positions[pair_second]

    return {
        "pair_first": pair_first,
        "pair_second": pair_second,
        "pair_dist": torch.linalg.vector_norm(pair_coord, dim=1),
        "pair_coord": pair_coord,
    }


def load_dataset(args: argparse.Namespace) -> tuple[dict[str, torch.Tensor], str]:
    if args.dataset == "k_chain":
        return as_kchain_arrays(create_kchains(args.k)), f"k={args.k} k-chain pair"
    if args.dataset == "incompleteness":
        if args.counterexample == "all":
            pairs_by_name = create_all_incompleteness_pairs()
            environments = [environment for name in COUNTEREXAMPLE_NAMES for environment in pairs_by_name[name]]
            return as_padded_incompleteness_arrays(environments), "all incompleteness counterexamples"
        return (
            as_incompleteness_arrays(create_incompleteness_pair(args.counterexample)),
            f"{args.counterexample} incompleteness pair",
        )
    raise ValueError(f"Unknown dataset {args.dataset!r}.")


def make_model(args: argparse.Namespace) -> torch.nn.Module:
    from hippynn.graphs import GraphModule, IdxType, inputs, networks, targets
    from hippynn.graphs.nodes.base import InputNode
    from hippynn.graphs.nodes.indexers import acquire_encoding_padding

    neighborhood_cutoff = getattr(args, "neighborhood_cutoff", "cutoff")
    if neighborhood_cutoff not in {"cutoff", "edges"}:
        raise ValueError(f"Unknown neighborhood cutoff {neighborhood_cutoff!r}. Expected 'cutoff' or 'edges'.")

    dist_hard_max = EDGE_NEIGHBORHOOD_DIST_HARD_MAX if neighborhood_cutoff == "edges" else args.dist_hard_max
    if args.dist_soft_max is not None:
        dist_soft_max = args.dist_soft_max
    elif neighborhood_cutoff == "edges":
        dist_soft_max = 6.0
    else:
        dist_soft_max = 6.0 if args.dist_hard_max <= 6.5 else 0.85 * args.dist_hard_max

    network_params = {
        "possible_species": [0, 1],
        "n_features": args.n_features,
        "n_sensitivities": args.n_sensitivities,
        "dist_soft_min": args.dist_soft_min,
        "dist_soft_max": dist_soft_max,
        "dist_hard_max": dist_hard_max,
        "n_interaction_layers": args.n_interaction_layers,
        "n_atom_layers": args.n_atom_layers,
    }
    network_class = networks.Hipnn
    if args.model == "hipnnvec":
        network_class = networks.HipnnVec
    elif args.model == "hiphop":
        network_class = networks.HipHopnn
        network_params.update(
            {
                "l_max": args.l_max,
                "n_max": args.n_max,
            }
        )

    species = inputs.SpeciesNode(db_name="Z")
    graph_inputs: list[object]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="HIP-HOP-NN is still in a beta state.*")
        if neighborhood_cutoff == "edges":
            _encoder, atom_indexer = acquire_encoding_padding(species, network_params["possible_species"])
            pair_first = InputNode(db_name="pair_first", index_state=IdxType.Pairs)
            pair_second = InputNode(db_name="pair_second", index_state=IdxType.Pairs)
            pair_dist = InputNode(db_name="pair_dist", index_state=IdxType.Pairs)
            network_parents = [
                atom_indexer.indexed_features,
                pair_first,
                pair_second,
                pair_dist,
            ]
            graph_inputs = [species, pair_first, pair_second, pair_dist]
            if args.model in {"hipnnvec", "hiphop"}:
                pair_coord = InputNode(db_name="pair_coord", index_state=IdxType.Pairs)
                network_parents.append(pair_coord)
                graph_inputs.append(pair_coord)
            network = network_class("geometric_model", tuple(network_parents), module_kwargs=network_params)
        else:
            positions = inputs.PositionsNode(db_name="R")
            network = network_class("geometric_model", (species, positions), module_kwargs=network_params)
            graph_inputs = [species, positions]

    logit = targets.HEnergyNode("logit", network, db_name="T")
    return GraphModule(graph_inputs, [logit.system_energy])


def model_forward_args(args: argparse.Namespace, arrays: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    neighborhood_cutoff = getattr(args, "neighborhood_cutoff", "cutoff")
    if neighborhood_cutoff not in {"cutoff", "edges"}:
        raise ValueError(f"Unknown neighborhood cutoff {neighborhood_cutoff!r}. Expected 'cutoff' or 'edges'.")

    if neighborhood_cutoff == "edges":
        pairs = edge_pair_tensors(arrays)
        inputs = [
            arrays["Z"],
            pairs["pair_first"],
            pairs["pair_second"],
            pairs["pair_dist"],
        ]
        if args.model in {"hipnnvec", "hiphop"}:
            inputs.append(pairs["pair_coord"])
    else:
        inputs = [arrays["Z"], arrays["R"]]

    return tuple(inputs)


def train(args: argparse.Namespace) -> dict[str, object]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    arrays, _ = load_dataset(args)
    targets = arrays["T"]
    forward_args = model_forward_args(args, arrays)

    model = make_model(args)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    final_loss = None
    final_accuracy = None
    final_margin_accuracy = None
    final_logits = None
    final_epoch = None

    for epoch in range(1, args.epochs + 1):
        (logits,) = model(*forward_args)
        loss = loss_fn(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            predictions = (logits >= 0).to(targets.dtype)
            accuracy = float((predictions == targets).to(torch.float32).mean().item())
            signed_targets = targets.mul(2).sub(1)
            margin_accuracy = float((signed_targets * logits >= args.success_margin).to(torch.float32).mean().item())

        final_loss = float(loss.item())
        final_accuracy = accuracy
        final_margin_accuracy = margin_accuracy
        final_logits = logits.detach().squeeze(-1)
        final_epoch = epoch

        if margin_accuracy >= args.stop_at_accuracy:
            break

    result = {
        "epoch": final_epoch,
        "loss": final_loss,
        "accuracy": final_accuracy,
        "margin_accuracy": final_margin_accuracy,
        "logits": final_logits.tolist(),
    }

    return result


def main() -> None:
    args = parse_args()
    result = train(args)
    print(
        f"epoch {result['epoch']} | loss {result['loss']:.6f} | "
        f"acc {result['accuracy']:.3f} | margin_acc {result['margin_accuracy']:.3f} | "
        f"logits {result['logits']}"
    )


if __name__ == "__main__":
    main()
