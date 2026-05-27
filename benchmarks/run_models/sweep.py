"""Sweep small HIP-NN configs on two-sample geometric benchmark tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

BENCHMARKS_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_ROOT))

import torch

from incompleteness.generate_data.incompleteness import COUNTEREXAMPLE_NAMES
from run_models.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["k_chain", "incompleteness"], default="k_chain")
    parser.add_argument("--k", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--counterexamples", choices=COUNTEREXAMPLE_NAMES, nargs="+", default=list(COUNTEREXAMPLE_NAMES))
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--model", choices=["hipnn", "hipnnvec", "hiphop"], default="hiphop")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 2])
    parser.add_argument("--interaction-layers", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--hard-cutoffs", type=float, nargs="+", default=[5.0, 10.0, 14.0, 18.0])
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--n-atom-layers", type=int, default=2)
    parser.add_argument("--n-features", type=int, default=32)
    parser.add_argument("--n-sensitivities", type=int, default=32)
    parser.add_argument("--dist-soft-min", type=float, default=1.0)
    parser.add_argument("--l-max", type=int, default=2)
    parser.add_argument("--n-max", type=int, default=3)
    parser.add_argument("--success-margin", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--no-progress", action="store_true", help="Hide per-run progress messages.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)

    dataset_items = args.k if args.dataset == "k_chain" else args.counterexamples
    total_runs = len(dataset_items) * len(args.hard_cutoffs) * len(args.interaction_layers) * len(args.seeds)
    run_index = 0

    if args.dataset == "k_chain":
        print(f"Sweeping {args.model} on k-chain k={args.k} with seeds={args.seeds}", flush=True)
        item_header = "k"
    else:
        print(f"Sweeping {args.model} on incompleteness {args.counterexamples} with seeds={args.seeds}", flush=True)
        item_header = "counterexample"
    print(f"Using params l-max: {args.l_max} and n-max: {args.n_max}", flush=True)
    print(f"Running {total_runs} trainings: {args.epochs} epochs max each", flush=True)
    print(f"success requires correct signs with logit margin >= {args.success_margin}", flush=True)
    print(f"successes/trials | {item_header} | hard cutoff | layers | final accuracies | margin accuracies | final logits", flush=True)

    for dataset_item in dataset_items:
        for hard_cutoff in args.hard_cutoffs:
            for n_layers in args.interaction_layers:
                results = []
                for seed in args.seeds:
                    run_index += 1
                    if not args.no_progress:
                        print(
                            f"running {run_index}/{total_runs}: "
                            f"{item_header}={dataset_item}, cutoff={hard_cutoff:g}, layers={n_layers}, seed={seed}",
                            flush=True,
                        )
                    train_args = SimpleNamespace(
                        dataset=args.dataset,
                        k=dataset_item if args.dataset == "k_chain" else args.k[0],
                        counterexample=dataset_item if args.dataset == "incompleteness" else args.counterexamples[0],
                        epochs=args.epochs,
                        seed=seed,
                        model=args.model,
                        learning_rate=args.learning_rate,
                        n_interaction_layers=n_layers,
                        n_atom_layers=args.n_atom_layers,
                        n_features=args.n_features,
                        n_sensitivities=args.n_sensitivities,
                        dist_soft_min=args.dist_soft_min,
                        dist_soft_max=6.0 if hard_cutoff <= 6.5 else 0.85 * hard_cutoff,
                        dist_hard_max=hard_cutoff,
                        l_max=args.l_max,
                        n_max=args.n_max,
                        log_every=args.log_every,
                        stop_at_accuracy=1.0,
                        success_margin=args.success_margin,
                        quiet=True,
                    )
                    results.append(train(train_args))

                successes = sum(result["margin_accuracy"] >= 1.0 for result in results)
                accuracies = [round(result["accuracy"], 3) for result in results]
                margin_accuracies = [round(result["margin_accuracy"], 3) for result in results]
                logits = [[round(value, 3) for value in result["logits"]] for result in results]
                item_text = f"{dataset_item:2d}" if args.dataset == "k_chain" else f"{dataset_item:22s}"
                print(
                    f"{successes}/{len(results)} | {item_text} | {hard_cutoff:10.2f} | {n_layers:6d} | "
                    f"{accuracies} | {margin_accuracies} | {logits}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
