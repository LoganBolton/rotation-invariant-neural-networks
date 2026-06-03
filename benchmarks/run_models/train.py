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
    parser.add_argument(
        "--coordinate-set",
        choices=("v2", "original"),
        default="v2",
        help="Coordinate set for --dataset incompleteness.",
    )
    parser.add_argument("--epochs", type=int, default=4000, help="Number of full-batch training epochs.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--model", choices=["hipnn", "hipnnvec", "hiphop"], default="hipnn", help="Network architecture to train.")
    parser.add_argument(
        "--readout",
        choices=["system", "central"],
        default="system",
        help="Use the normal system-summed readout or a central-atom-only readout.",
    )
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
    parser.add_argument("--log-every", type=int, default=250, help="Print progress every N epochs.")
    parser.add_argument("--stop-at-accuracy", type=float, default=1.0, help="Early-stop once this margin accuracy is reached.")
    parser.add_argument("--success-margin", type=float, default=0.1, help="Report margin accuracy using this logit margin.")
    parser.add_argument("--quiet", action="store_true", help="Only print the final result.")
    return parser.parse_args()


def resolve_dist_hard_max(args: argparse.Namespace) -> float:
    if resolve_neighborhood_cutoff(args) == "edges":
        return EDGE_NEIGHBORHOOD_DIST_HARD_MAX
    return args.dist_hard_max


def resolve_dist_soft_max(args: argparse.Namespace) -> float:
    if args.dist_soft_max is not None:
        return args.dist_soft_max
    if resolve_neighborhood_cutoff(args) == "edges":
        return 6.0
    return 6.0 if args.dist_hard_max <= 6.5 else 0.85 * args.dist_hard_max


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def requires_pair_coord(args: argparse.Namespace) -> bool:
    return args.model in {"hipnnvec", "hiphop"}


def resolve_neighborhood_cutoff(args: argparse.Namespace) -> str:
    value = getattr(args, "neighborhood_cutoff", "cutoff")
    if value not in {"cutoff", "edges"}:
        raise ValueError(f"Unknown neighborhood cutoff {value!r}. Expected 'cutoff' or 'edges'.")
    return value


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

    real_atom_mask = species != 0
    real_flat_indices = torch.nonzero(real_atom_mask.reshape(-1), as_tuple=False).squeeze(1)
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
        coordinate_set = getattr(args, "coordinate_set", "v2")
        if args.counterexample == "all":
            pairs_by_name = create_all_incompleteness_pairs(coordinate_set=coordinate_set)
            environments = [environment for name in COUNTEREXAMPLE_NAMES for environment in pairs_by_name[name]]
            return as_padded_incompleteness_arrays(environments), f"all {coordinate_set} incompleteness counterexamples"
        return (
            as_incompleteness_arrays(create_incompleteness_pair(args.counterexample, coordinate_set=coordinate_set)),
            f"{args.counterexample} {coordinate_set} incompleteness pair",
        )
    raise ValueError(f"Unknown dataset {args.dataset!r}.")


def make_model(args: argparse.Namespace) -> torch.nn.Module:
    from hippynn.graphs import GraphModule, IdxType, inputs, networks, targets
    from hippynn.graphs.indextypes import index_type_coercion
    from hippynn.graphs.nodes.base import InputNode
    from hippynn.graphs.nodes.indexers import acquire_encoding_padding
    from hippynn.graphs.nodes.tags import AtomIndexer

    neighborhood_cutoff = resolve_neighborhood_cutoff(args)

    dist_soft_max = resolve_dist_soft_max(args)
    network_params = {
        "possible_species": [0, 1],
        "n_features": args.n_features,
        "n_sensitivities": args.n_sensitivities,
        "dist_soft_min": args.dist_soft_min,
        "dist_soft_max": dist_soft_max,
        "dist_hard_max": resolve_dist_hard_max(args),
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
            if requires_pair_coord(args):
                pair_coord = InputNode(db_name="pair_coord", index_state=IdxType.Pairs)
                network_parents.append(pair_coord)
                graph_inputs.append(pair_coord)
            network = network_class("geometric_model", tuple(network_parents), module_kwargs=network_params)
        else:
            positions = inputs.PositionsNode(db_name="R")
            network = network_class("geometric_model", (species, positions), module_kwargs=network_params)
            graph_inputs = [species, positions]

    readout = getattr(args, "readout", "system")
    if readout == "central":
        central_atom_mask_input = InputNode(db_name="central_atom_mask", index_state=IdxType.SysAtom)
        central_atom_mask = index_type_coercion(central_atom_mask_input, IdxType.Atoms, hints=(network,))
        atom_indexer = network.find_unique_relative(AtomIndexer)
        logit = targets.HEnergyNode(
            "central_logit",
            (network, atom_indexer.system_index, atom_indexer.n_systems, central_atom_mask),
            module_kwargs={"feature_sizes": network.torch_module.feature_sizes},
            db_name="T",
        )
        return GraphModule([*graph_inputs, central_atom_mask_input], [logit.system_energy])

    logit = targets.HEnergyNode("logit", network, db_name="T")
    return GraphModule(graph_inputs, [logit.system_energy])


def model_forward_args(args: argparse.Namespace, arrays: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    neighborhood_cutoff = resolve_neighborhood_cutoff(args)
    readout = getattr(args, "readout", "system")

    if neighborhood_cutoff == "edges":
        pairs = edge_pair_tensors(arrays)
        inputs = [
            arrays["Z"],
            pairs["pair_first"],
            pairs["pair_second"],
            pairs["pair_dist"],
        ]
        if requires_pair_coord(args):
            inputs.append(pairs["pair_coord"])
    elif neighborhood_cutoff == "cutoff":
        inputs = [arrays["Z"], arrays["R"]]

    if readout == "central":
        central_atom_mask = arrays.get("central_atom_mask")
        if central_atom_mask is None:
            raise ValueError("Central readout requires the dataset arrays to include 'central_atom_mask'.")
        inputs.append(central_atom_mask)

    return tuple(inputs)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = (logits >= 0).to(targets.dtype)
    return float((predictions == targets).to(torch.float32).mean().item())


def margin_accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor, margin: float) -> float:
    signed_targets = targets.mul(2).sub(1)
    return float((signed_targets * logits >= margin).to(torch.float32).mean().item())


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)

    arrays, dataset_description = load_dataset(args)
    species = arrays["Z"]
    positions = arrays["R"]
    targets = arrays["T"]
    readout = getattr(args, "readout", "system")
    neighborhood_cutoff = resolve_neighborhood_cutoff(args)
    if neighborhood_cutoff == "edges" and args.dataset != "incompleteness":
        raise ValueError("Dataset-defined edges are only available for the incompleteness dataset.")
    central_atom_mask = arrays.get("central_atom_mask")
    if readout == "central" and central_atom_mask is None:
        raise ValueError("Central readout requires the dataset arrays to include 'central_atom_mask'.")
    forward_args = model_forward_args(args, arrays)

    model = make_model(args)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if not args.quiet:
        print(
            f"Training {args.model} with {readout} readout and {neighborhood_cutoff} neighborhood "
            f"on {dataset_description}"
        )
        print(f"Z: {tuple(species.shape)} {species.dtype}; R: {tuple(positions.shape)} {positions.dtype}; T: {targets.squeeze(-1).tolist()}")
        if readout == "central":
            print(f"central_atom_mask: {central_atom_mask.tolist()}")
        if neighborhood_cutoff == "edges":
            print(f"edges: {arrays['edge_index'].shape[1]}")
        print(
            "Network: "
            f"{args.n_interaction_layers} interactions, "
            f"{args.n_atom_layers} atom layers, "
            f"{args.n_features} features, cutoff {resolve_dist_hard_max(args)}, soft max {resolve_dist_soft_max(args)}"
        )
        if args.model == "hiphop":
            print(f"HIP-HOP tensors: l_max={args.l_max}, n_max={args.n_max}")

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
            accuracy = accuracy_from_logits(logits, targets)
            margin_accuracy = margin_accuracy_from_logits(logits, targets, args.success_margin)

        final_loss = float(loss.item())
        final_accuracy = accuracy
        final_margin_accuracy = margin_accuracy
        final_logits = logits.detach().squeeze(-1)
        final_epoch = epoch

        should_log = epoch == 1 or epoch % args.log_every == 0 or margin_accuracy >= args.stop_at_accuracy
        if should_log and not args.quiet:
            probs = torch.sigmoid(final_logits)
            print(
                f"epoch {epoch:5d} | loss {final_loss:.6f} | acc {accuracy:.3f} | "
                f"margin_acc {margin_accuracy:.3f} | "
                f"logits {final_logits.tolist()} | probs {probs.tolist()}"
            )

        if margin_accuracy >= args.stop_at_accuracy:
            if not args.quiet:
                print(f"Reached margin accuracy {margin_accuracy:.3f}; stopping at epoch {epoch}.")
            break

    result = {
        "epoch": final_epoch,
        "loss": final_loss,
        "accuracy": final_accuracy,
        "margin_accuracy": final_margin_accuracy,
        "logits": final_logits.tolist(),
    }

    if not args.quiet:
        print("Final:")
        print(f"  loss: {final_loss:.6f}")
        print(f"  accuracy: {final_accuracy:.3f}")
        print(f"  margin accuracy @ {args.success_margin}: {final_margin_accuracy:.3f}")
        print(f"  logits: {final_logits.tolist()}")

    return result


def main() -> None:
    args = parse_args()
    result = train(args)
    if args.quiet:
        print(
            f"epoch {result['epoch']} | loss {result['loss']:.6f} | "
            f"acc {result['accuracy']:.3f} | margin_acc {result['margin_accuracy']:.3f} | "
            f"logits {result['logits']}"
        )


if __name__ == "__main__":
    main()
