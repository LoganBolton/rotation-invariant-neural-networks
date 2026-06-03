"""Sweep small HIP-NN configs on two-sample geometric benchmark tasks.

Example run: 
`uv run python benchmarks/run_models/sweep.py --dataset incompleteness --hard-cutoffs 5 10 14 --model-configs default --output-dir benchmarks/incompleteness/results/central_node`
uv run python benchmarks/run_models/sweep.py --dataset k_chain --hard-cutoffs 5 10 14 --model-configs default --output-dir benchmarks/k_chain/results/central_node
"""



from __future__ import annotations

import argparse
import contextlib
import concurrent.futures
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TextIO

BENCHMARKS_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_ROOT))

import torch

from incompleteness.generate_data.incompleteness import COUNTEREXAMPLE_NAMES
from run_models.train import train

# Specify model type, max_l, max_n
DEFAULT_MODEL_CONFIGS = (
    ("hipnn", 0, 1),
    ("hiphop", 1, 2),
    ("hiphop", 2, 2),
    ("hiphop", 2, 3),
    ("hiphop", 3, 2),
    ("hiphop", 3, 4),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["k_chain", "incompleteness"], default="k_chain")
    parser.add_argument("--k", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--counterexamples", choices=COUNTEREXAMPLE_NAMES, nargs="+", default=list(COUNTEREXAMPLE_NAMES))
    parser.add_argument("--coordinate-set", choices=("v2", "original"), default="v2")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--model", choices=["hipnn", "hipnnvec", "hiphop"], default="hiphop")
    parser.add_argument("--readout", choices=["system", "central"], default="central")
    parser.add_argument("--neighborhood-cutoff", choices=["cutoff", "edges"], default="cutoff")
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
    parser.add_argument(
        "--model-configs",
        nargs="+",
        default=None,
        help=(
            "Run multiple configs and write one log per config. "
            "Use 'default' for hipnn plus the standard HIP-HOP l/n configs, "
            "or values like 'hipnn', 'l2_n3', or 'hiphop:2:3'."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for per-config markdown logs when --model-configs is used.",
    )
    parser.add_argument(
        "--parallel-configs",
        type=int,
        default=None,
        help="Number of model configs to train concurrently. Defaults to all configs.",
    )
    parser.add_argument("--success-margin", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--no-progress", action="store_true", help="Hide per-run progress messages.")
    return parser.parse_args()


def parse_model_configs(configs: list[str] | None) -> list[tuple[str, int, int]]:
    if configs is None:
        return []
    if configs == ["default"]:
        return list(DEFAULT_MODEL_CONFIGS)

    parsed = []
    for config in configs:
        if config == "default":
            parsed.extend(DEFAULT_MODEL_CONFIGS)
        elif config == "hipnn":
            parsed.append(("hipnn", 0, 1))
        elif config.startswith("l") and "_n" in config:
            l_text, n_text = config.removeprefix("l").split("_n", maxsplit=1)
            parsed.append(("hiphop", int(l_text), int(n_text)))
        elif config.startswith("hiphop:"):
            _model, l_text, n_text = config.split(":", maxsplit=2)
            parsed.append(("hiphop", int(l_text), int(n_text)))
        else:
            raise ValueError(
                f"Unknown model config {config!r}. Use 'default', 'hipnn', 'l2_n3', or 'hiphop:2:3'."
            )

    return parsed


def config_log_name(model: str, l_max: int, n_max: int) -> str:
    if model == "hipnn":
        return "l0_n1.md"
    return f"l{l_max}_n{n_max}.md"


def args_for_config(args: argparse.Namespace, model: str, l_max: int, n_max: int) -> argparse.Namespace:
    config_args = argparse.Namespace(**vars(args))
    config_args.model = model
    config_args.l_max = l_max
    config_args.n_max = n_max
    return config_args


def run_sweep(args: argparse.Namespace, output: TextIO) -> None:
    torch.set_num_threads(1)

    dataset_items = args.k if args.dataset == "k_chain" else args.counterexamples
    total_runs = len(dataset_items) * len(args.hard_cutoffs) * len(args.interaction_layers) * len(args.seeds)
    run_index = 0

    if args.dataset == "k_chain":
        print(
            f"Sweeping {args.model} with {args.readout} readout and {args.neighborhood_cutoff} neighborhood "
            f"on k-chain k={args.k} with seeds={args.seeds}",
            flush=True,
            file=output,
        )
        item_header = "k"
    else:
        print(
            f"Sweeping {args.model} with {args.readout} readout and {args.neighborhood_cutoff} neighborhood "
            f"on {args.coordinate_set} incompleteness {args.counterexamples} with seeds={args.seeds}",
            flush=True,
            file=output,
        )
        item_header = "counterexample"
    print(f"Using params l-max: {args.l_max} and n-max: {args.n_max}", flush=True, file=output)
    print(f"Running {total_runs} trainings: {args.epochs} epochs max each", flush=True, file=output)
    print(f"success requires correct signs with logit margin >= {args.success_margin}", flush=True, file=output)
    print(
        f"successes/trials | {item_header} | hard cutoff | layers | final accuracies | margin accuracies | final logits",
        flush=True,
        file=output,
    )

    for dataset_item in dataset_items:
        for hard_cutoff in args.hard_cutoffs:
            for n_layers in args.interaction_layers:
                results = []
                for seed in args.seeds:
                    run_index += 1
                    train_args = SimpleNamespace(
                        dataset=args.dataset,
                        k=dataset_item if args.dataset == "k_chain" else args.k[0],
                        counterexample=dataset_item if args.dataset == "incompleteness" else args.counterexamples[0],
                        coordinate_set=args.coordinate_set,
                        epochs=args.epochs,
                        seed=seed,
                        model=args.model,
                        readout=args.readout,
                        neighborhood_cutoff=args.neighborhood_cutoff,
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
                    file=output,
                )


def run_config_to_file(args: argparse.Namespace, model: str, l_max: int, n_max: int, output_dir: Path) -> Path:
    config_args = args_for_config(args, model, l_max, n_max)
    output_file = output_dir / config_log_name(model, l_max, n_max)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as output:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            run_sweep(config_args, output)
    return output_file


def run_model_config_batch(args: argparse.Namespace) -> None:
    configs = parse_model_configs(args.model_configs)
    if not configs:
        raise ValueError("--model-configs was provided but no configs were parsed.")
    if args.output_dir is None:
        raise ValueError("--output-dir is required with --model-configs.")

    max_workers = args.parallel_configs or len(configs)
    if max_workers < 1:
        raise ValueError("--parallel-configs must be at least 1.")

    print(f"Running {len(configs)} model configs with {max_workers} parallel workers.", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_config_to_file, args, model, l_max, n_max, args.output_dir)
            for model, l_max, n_max in configs
        ]
        for future in concurrent.futures.as_completed(futures):
            output_file = future.result()
            print(f"Saved sweep log: {output_file}", flush=True)


def main() -> None:
    args = parse_args()
    if args.model_configs is not None:
        run_model_config_batch(args)
        return

    run_sweep(args, sys.stdout)


if __name__ == "__main__":
    main()
