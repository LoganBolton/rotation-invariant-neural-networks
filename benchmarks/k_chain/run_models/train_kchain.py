"""Train HIP-NN on the two-sample k-chain distinguishability task."""

from __future__ import annotations

import argparse
import os
import random
import sys
import warnings
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from generate_data.kchains import as_hippynn_arrays, create_kchains


os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib-cache")))
os.environ.setdefault("HIPPYNN_USE_CUSTOM_KERNELS", "False")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=4, help="Number of middle chain nodes.")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of full-batch training epochs.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--model", choices=["hipnn", "hipnnvec", "hiphop"], default="hipnn", help="Network architecture to train.")
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


def resolve_dist_soft_max(args: argparse.Namespace) -> float:
    if args.dist_soft_max is not None:
        return args.dist_soft_max
    return 6.0 if args.dist_hard_max <= 6.5 else 0.85 * args.dist_hard_max


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def make_model(args: argparse.Namespace) -> torch.nn.Module:
    from hippynn.graphs import GraphModule, inputs, networks, targets

    dist_soft_max = resolve_dist_soft_max(args)
    network_params = {
        "possible_species": [0, 1],
        "n_features": args.n_features,
        "n_sensitivities": args.n_sensitivities,
        "dist_soft_min": args.dist_soft_min,
        "dist_soft_max": dist_soft_max,
        "dist_hard_max": args.dist_hard_max,
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
    positions = inputs.PositionsNode(db_name="R")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="HIP-HOP-NN is still in a beta state.*")
        network = network_class("geometric_model", (species, positions), module_kwargs=network_params)
    logit = targets.HEnergyNode("logit", network, db_name="T")
    return GraphModule([species, positions], [logit.system_energy])


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = (logits >= 0).to(targets.dtype)
    return float((predictions == targets).to(torch.float32).mean().item())


def margin_accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor, margin: float) -> float:
    signed_targets = targets.mul(2).sub(1)
    return float((signed_targets * logits >= margin).to(torch.float32).mean().item())


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)

    arrays = as_hippynn_arrays(create_kchains(args.k))
    species = arrays["Z"]
    positions = arrays["R"]
    targets = arrays["T"]

    model = make_model(args)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if not args.quiet:
        print(f"Training {args.model} on k={args.k} k-chain pair")
        print(f"Z: {tuple(species.shape)} {species.dtype}; R: {tuple(positions.shape)} {positions.dtype}; T: {targets.squeeze(-1).tolist()}")
        print(
            "Network: "
            f"{args.n_interaction_layers} interactions, "
            f"{args.n_atom_layers} atom layers, "
            f"{args.n_features} features, cutoff {args.dist_hard_max}, soft max {resolve_dist_soft_max(args)}"
        )
        if args.model == "hiphop":
            print(f"HIP-HOP tensors: l_max={args.l_max}, n_max={args.n_max}")

    final_loss = None
    final_accuracy = None
    final_margin_accuracy = None
    final_logits = None
    final_epoch = None

    for epoch in range(1, args.epochs + 1):
        (logits,) = model(species, positions)
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
