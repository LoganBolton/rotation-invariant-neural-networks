"""Sweep small HIP-NN configs on two-sample geometric benchmark tasks."""

from __future__ import annotations

import argparse
import contextlib
import concurrent.futures
import pprint
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TextIO

BENCHMARKS_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_ROOT))

import torch

from incompleteness.generate_data.incompleteness import COUNTEREXAMPLE_NAMES
from run_models.sweep_configs import (
    args_for_config,
    args_for_ring_graph_config,
    config_log_name,
    parse_model_configs,
    parse_ring_graph_configs,
)
from run_models.sweep_results import (
    make_run_result_record,
    make_sweep_result_record,
    write_json_index,
    write_result_json,
)
from run_models.train import train

DEFAULT_DIST_SOFT_MIN = 1.0
ROTATING_RING_DIST_SOFT_MIN = 0.5


def default_dist_soft_min(dataset: str) -> float:
    return ROTATING_RING_DIST_SOFT_MIN if dataset == "rotating_ring" else DEFAULT_DIST_SOFT_MIN


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.dist_soft_min is None:
        args.dist_soft_min = default_dist_soft_min(args.dataset)
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["k_chain", "incompleteness", "rotating_ring"], default="k_chain")
    parser.add_argument("--k", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--counterexamples", choices=COUNTEREXAMPLE_NAMES, nargs="+", default=list(COUNTEREXAMPLE_NAMES))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--model", choices=["hipnn", "hipnnvec", "hiphop"], default="hiphop")
    parser.add_argument("--neighborhood-cutoff", choices=["cutoff", "edges"], default="cutoff")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 2])
    parser.add_argument("--interaction-layers", type=int, nargs="+", default=[1])
    parser.add_argument("--hard-cutoffs", type=float, nargs="+", default=[4.0])
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--n-atom-layers", type=int, default=2)
    parser.add_argument("--n-features", type=int, default=32)
    parser.add_argument("--n-sensitivities", type=int, default=32)
    parser.add_argument("--device", default="cpu", help="Torch device for training, e.g. 'cpu', 'cuda', or 'cuda:0'.")
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="Optional list of devices to round-robin across parallel sweep jobs, e.g. cuda:0 cuda:1.",
    )
    parser.add_argument(
        "--dist-soft-min",
        type=float,
        default=None,
        help="Sensitivity soft minimum. Defaults to 0.5 for rotating_ring and 1.0 otherwise.",
    )
    parser.add_argument("--l-max", type=int, default=2)
    parser.add_argument("--n-max", type=int, default=3)
    parser.add_argument(
        "--hiphop-group-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable HIP-HOP's per-invariant GroupNorm (use --no-hiphop-group-norm to disable it).",
    )
    parser.add_argument("--ring-n-graphs", type=int, default=2, help="Number of rotating-ring graphs to generate.")
    parser.add_argument("--ring-seed", type=int, default=0, help="Dataset seed for rotating-ring generation.")
    parser.add_argument("--ring-n-inner", type=int, default=3, help="Number of rotating-ring inner nodes.")
    parser.add_argument("--ring-n-outer", type=int, default=3, help="Number of rotating-ring outer nodes.")
    parser.add_argument("--ring-inner-radius", type=float, default=1.0, help="Rotating-ring inner radius.")
    parser.add_argument(
        "--ring-z-phase-sample",
        action="store_true",
        help="Train on the two-graph equal-radius inner-z-rotation sample.",
    )
    parser.add_argument("--ring-z-phase-far-inner-rotation-deg", type=float, default=15.0)
    parser.add_argument(
        "--ring-outer-gap",
        type=float,
        default=1.2,
        help="Rotating-ring outer radius minus inner radius.",
    )
    parser.add_argument(
        "--ring-outer-3d-rotation-deg",
        type=float,
        default=0.0,
        help="Maximum out-of-plane outer-ring tilt in degrees for rotating-ring generation.",
    )
    parser.add_argument(
        "--ring-outer-3d-axis-deg",
        type=float,
        default=0.0,
        help="In-plane direction of the rotating-ring outer tilt axis in degrees.",
    )
    parser.add_argument(
        "--model-configs",
        nargs="+",
        default=None,
        help=(
            "Run multiple configs and write one log per config. "
            "Use 'default' or 'all' for the standard HIP-HOP l/n configs, "
            "or values like 'hipnn', 'l2_n3', or 'hiphop:2:3'."
        ),
    )
    parser.add_argument(
        "--ring-graph-configs",
        nargs="+",
        default=None,
        help=(
            "Rotating-ring graph versions to sweep. Use 'all_2d', 'all_3d', or 'all', "
            "or explicit values like '2d_3inner_4_outer', '3d_4inner_4_outer', or '2d:3:4'. "
            "When --output-dir already contains matching graph-version folders, all_* discovers them."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for per-config markdown logs when --model-configs is used.",
    )
    parser.add_argument(
        "--results-json-dir",
        type=Path,
        default=None,
        help=(
            "Directory for machine-readable JSON result files. "
            "Defaults to <output-dir>/json_results when --output-dir is set, "
            "or sweep_json_results for a direct stdout sweep."
        ),
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


def format_result_logits(args: argparse.Namespace, results: list[dict[str, object]]) -> object:
    if args.dataset != "rotating_ring":
        return [[round(value, 3) for value in result["logits"]] for result in results]

    summaries = []
    for result in results:
        logits = result["logits"]
        midpoint = len(logits) // 2
        by_class = []
        for label, class_logits in enumerate((logits[:midpoint], logits[midpoint:])):
            if class_logits:
                mean = sum(class_logits) / len(class_logits)
                minimum = min(class_logits)
                maximum = max(class_logits)
            else:
                mean = minimum = maximum = float("nan")
            by_class.append(
                {
                    "label": label,
                    "mean": round(mean, 3),
                    "min": round(minimum, 3),
                    "max": round(maximum, 3),
                }
            )
        summaries.append(by_class)
    return summaries


def run_sweep(args: argparse.Namespace, output: TextIO) -> None:
    normalize_args(args)
    torch.set_num_threads(1)

    if args.dataset == "k_chain":
        dataset_items = list(args.k)
    elif args.dataset == "incompleteness":
        dataset_items = list(args.counterexamples)
    else:
        dataset_items = ["rotating_ring"]
    total_runs = len(dataset_items) * len(args.hard_cutoffs) * len(args.interaction_layers) * len(args.seeds)
    run_index = 0
    run_records: list[dict[str, object]] = []
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}

    print("Config:", flush=True, file=output)
    print(pprint.pformat(config, sort_dicts=True), flush=True, file=output)
    print("", flush=True, file=output)

    if args.dataset == "k_chain":
        print(
            f"Sweeping {args.model} with {args.neighborhood_cutoff} neighborhood "
            f"on k-chain k={args.k} with seeds={args.seeds}",
            flush=True,
            file=output,
        )
        item_header = "k"
    elif args.dataset == "incompleteness":
        print(
            f"Sweeping {args.model} with {args.neighborhood_cutoff} neighborhood "
            f"on incompleteness {args.counterexamples} with seeds={args.seeds}",
            flush=True,
            file=output,
        )
        item_header = "counterexample"
    else:
        print(
            f"Sweeping {args.model} with {args.neighborhood_cutoff} neighborhood "
            f"on rotating-ring n_graphs={args.ring_n_graphs} with seeds={args.seeds}",
            flush=True,
            file=output,
        )
        print(
            f"Ring geometry: inner_radius={args.ring_inner_radius:g}, "
            f"outer_gap={args.ring_outer_gap:g}, "
            f"outer_radius={args.ring_inner_radius + args.ring_outer_gap:g}",
            flush=True,
            file=output,
        )
        item_header = "dataset"
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
            dist_soft_max = None if args.neighborhood_cutoff == "edges" else 6.0 if hard_cutoff <= 6.5 else 0.85 * hard_cutoff
            for n_layers in args.interaction_layers:
                results = []
                for seed in args.seeds:
                    run_index += 1
                    train_args = SimpleNamespace(
                        dataset=args.dataset,
                        k=dataset_item if args.dataset == "k_chain" else args.k[0],
                        counterexample=dataset_item if args.dataset == "incompleteness" else args.counterexamples[0],
                        ring_n_graphs=args.ring_n_graphs,
                        ring_seed=args.ring_seed,
                        ring_n_inner=args.ring_n_inner,
                        ring_n_outer=args.ring_n_outer,
                        ring_inner_radius=args.ring_inner_radius,
                        ring_z_phase_sample=args.ring_z_phase_sample,
                        ring_z_phase_far_inner_rotation_deg=args.ring_z_phase_far_inner_rotation_deg,
                        ring_outer_gap=args.ring_outer_gap,
                        ring_outer_3d_rotation_deg=args.ring_outer_3d_rotation_deg,
                        ring_outer_3d_axis_deg=args.ring_outer_3d_axis_deg,
                        epochs=args.epochs,
                        seed=seed,
                        model=args.model,
                        neighborhood_cutoff=args.neighborhood_cutoff,
                        learning_rate=args.learning_rate,
                        n_interaction_layers=n_layers,
                        n_atom_layers=args.n_atom_layers,
                        n_features=args.n_features,
                        n_sensitivities=args.n_sensitivities,
                        device=args.device,
                        dist_soft_min=args.dist_soft_min,
                        dist_soft_max=dist_soft_max,
                        dist_hard_max=hard_cutoff,
                        l_max=args.l_max,
                        n_max=args.n_max,
                        hiphop_group_norm=args.hiphop_group_norm,
                        stop_at_accuracy=1.0,
                        success_margin=args.success_margin,
                    )
                    start_time = time.perf_counter()
                    result = train(train_args)
                    train_time = time.perf_counter() - start_time
                    results.append(result)
                    run_records.append(
                        make_run_result_record(
                            args,
                            dataset_item,
                            hard_cutoff,
                            n_layers,
                            seed,
                            dist_soft_max,
                            result,
                            train_time,
                        )
                    )

                successes = sum(result["margin_accuracy"] >= 1.0 for result in results)
                accuracies = [round(result["accuracy"], 3) for result in results]
                margin_accuracies = [round(result["margin_accuracy"], 3) for result in results]
                logits = format_result_logits(args, results)
                item_text = f"{dataset_item:2d}" if args.dataset == "k_chain" else f"{dataset_item:22s}"
                print(
                    f"{successes}/{len(results)} | {item_text} | {hard_cutoff:10.2f} | {n_layers:6d} | "
                    f"{accuracies} | {margin_accuracies} | {logits}",
                    flush=True,
                    file=output,
                )

    results_dir = args.results_json_dir
    if results_dir is None:
        results_dir = args.output_dir / "json_results" if args.output_dir else Path("sweep_json_results")
    result_path = results_dir / ("l0_n1.json" if args.model == "hipnn" else f"l{args.l_max}_n{args.n_max}.json")
    write_result_json(result_path, make_sweep_result_record(args, run_records))
    print(f"Saved sweep JSON: {result_path}", flush=True, file=output)


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

    graph_configs = parse_ring_graph_configs(args.ring_graph_configs, args.output_dir)
    if graph_configs and args.dataset != "rotating_ring":
        raise ValueError("--ring-graph-configs can only be used with --dataset rotating_ring.")

    config_jobs = []
    if graph_configs:
        base_json_dir = args.results_json_dir
        if base_json_dir is None:
            base_json_dir = args.output_dir / "json_results" if args.output_dir else Path("sweep_json_results")
        for graph_config in graph_configs:
            graph_output_dir = args.output_dir / graph_config.name
            graph_args = args_for_ring_graph_config(args, graph_config, graph_output_dir)
            graph_args.results_json_dir = base_json_dir / graph_config.name
            for model, l_max, n_max in configs:
                config_jobs.append((graph_args, model, l_max, n_max, graph_output_dir))
    else:
        for model, l_max, n_max in configs:
            config_jobs.append((args, model, l_max, n_max, args.output_dir))

    if args.devices:
        assigned_jobs = []
        for job_index, (job_args, model, l_max, n_max, output_dir) in enumerate(config_jobs):
            assigned_args = argparse.Namespace(**vars(job_args))
            assigned_args.device = args.devices[job_index % len(args.devices)]
            assigned_jobs.append((assigned_args, model, l_max, n_max, output_dir))
        config_jobs = assigned_jobs

    max_workers = args.parallel_configs or len(configs)
    if max_workers < 1:
        raise ValueError("--parallel-configs must be at least 1.")

    print(f"Running {len(config_jobs)} sweep jobs with {max_workers} parallel workers.", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_config_to_file, job_args, model, l_max, n_max, output_dir)
            for job_args, model, l_max, n_max, output_dir in config_jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            output_file = future.result()
            print(f"Saved sweep log: {output_file}", flush=True)

    results_dir = args.results_json_dir
    if results_dir is None:
        results_dir = args.output_dir / "json_results" if args.output_dir else Path("sweep_json_results")
    write_json_index(results_dir)
    print(f"Saved sweep JSON index: {results_dir / 'index.json'}", flush=True)


def main() -> None:
    args = normalize_args(parse_args())
    if args.ring_graph_configs is not None and args.output_dir is None:
        raise ValueError("--output-dir is required with --ring-graph-configs.")
    if args.model_configs is not None:
        run_model_config_batch(args)
        return
    if args.output_dir is not None:
        args.model_configs = ["default"]
        run_model_config_batch(args)
        return

    run_sweep(args, sys.stdout)


if __name__ == "__main__":
    main()
