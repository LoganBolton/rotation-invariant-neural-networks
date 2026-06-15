"""Sweep small HIP-NN configs on two-sample geometric benchmark tasks.

Example run:
`uv run python benchmarks/run_models/sweep.py --dataset incompleteness --hard-cutoffs 5 10 14 --model-configs default --output-dir benchmarks/incompleteness/results/system_node`
uv run python benchmarks/run_models/sweep.py --dataset k_chain --hard-cutoffs 5 10 14 --model-configs default --output-dir benchmarks/k_chain/results/system_node
uv run python benchmarks/run_models/sweep.py --dataset rotating_ring --neighborhood-cutoff edges --hard-cutoffs 4 5 --model-configs default --output-dir benchmarks/rotating_ring/results/system_original_edges
"""



from __future__ import annotations

import argparse
import contextlib
import concurrent.futures
import pprint
import re
import sys
from dataclasses import dataclass
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
    ("hiphop", 0, 1),
    ("hiphop", 1, 4),
    ("hiphop", 2, 4),
    ("hiphop", 3, 4),
)

DEFAULT_2D_RING_GRAPH_CONFIGS = (
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 2),
    (3, 3),
    (3, 4),
    (4, 3),
    (4, 4),
    (5, 5),
)
DEFAULT_3D_RING_GRAPH_CONFIGS = (
    (3, 3),
    (4, 4),
)
RING_GRAPH_CONFIG_RE = re.compile(r"^(?P<dimension>[23]d)_(?P<inner>\d+)inner_(?P<outer>\d+)_outer$")

# DEFAULT_MODEL_CONFIGS = (
#     ("hiphop", 0, 1),
#     ("hiphop", 1, 4),
#     ("hiphop", 2, 4),
#     ("hiphop", 3, 4)
# )
DEFAULT_DIST_SOFT_MIN = 1.0
ROTATING_RING_DIST_SOFT_MIN = 0.5


@dataclass(frozen=True)
class RingGraphConfig:
    name: str
    n_inner: int
    n_outer: int
    outer_3d_rotation_deg: float
    outer_3d_axis_deg: float


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
    parser.add_argument(
        "--dist-soft-min",
        type=float,
        default=None,
        help="Sensitivity soft minimum. Defaults to 0.5 for rotating_ring and 1.0 otherwise.",
    )
    parser.add_argument("--l-max", type=int, default=2)
    parser.add_argument("--n-max", type=int, default=3)
    parser.add_argument("--ring-n-graphs", type=int, default=100, help="Number of rotating-ring graphs to generate.")
    parser.add_argument("--ring-seed", type=int, default=0, help="Dataset seed for rotating-ring generation.")
    parser.add_argument("--ring-n-inner", type=int, default=3, help="Number of rotating-ring inner nodes.")
    parser.add_argument("--ring-n-outer", type=int, default=3, help="Number of rotating-ring outer nodes.")
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
    if configs in (["default"], ["all"]):
        return list(DEFAULT_MODEL_CONFIGS)

    parsed = []
    for config in configs:
        if config in {"default", "all"}:
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
                f"Unknown model config {config!r}. Use 'default', 'all', 'hipnn', 'l2_n3', or 'hiphop:2:3'."
            )

    return parsed


def ring_graph_config_name(dimension: str, n_inner: int, n_outer: int) -> str:
    return f"{dimension}_{n_inner}inner_{n_outer}_outer"


def make_ring_graph_config(dimension: str, n_inner: int, n_outer: int) -> RingGraphConfig:
    if dimension not in {"2d", "3d"}:
        raise ValueError(f"Unknown rotating-ring graph dimension {dimension!r}. Expected '2d' or '3d'.")

    outer_3d_rotation_deg = 360.0 if dimension == "3d" else 0.0
    outer_3d_axis_deg = 360.0 if dimension == "3d" else 0.0
    return RingGraphConfig(
        name=ring_graph_config_name(dimension, n_inner, n_outer),
        n_inner=n_inner,
        n_outer=n_outer,
        outer_3d_rotation_deg=outer_3d_rotation_deg,
        outer_3d_axis_deg=outer_3d_axis_deg,
    )


def discover_ring_graph_configs(output_dir: Path | None, dimension: str | None) -> list[RingGraphConfig]:
    if output_dir is None or not output_dir.exists():
        return []

    configs = []
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        match = RING_GRAPH_CONFIG_RE.match(child.name)
        if not match:
            continue
        child_dimension = match.group("dimension")
        if dimension is not None and child_dimension != dimension:
            continue
        configs.append(
            make_ring_graph_config(
                child_dimension,
                int(match.group("inner")),
                int(match.group("outer")),
            )
        )

    return sorted(configs, key=lambda config: (config.name.startswith("3d_"), config.n_inner, config.n_outer))


def default_ring_graph_configs(dimension: str | None) -> list[RingGraphConfig]:
    configs = []
    if dimension in {None, "2d"}:
        configs.extend(make_ring_graph_config("2d", n_inner, n_outer) for n_inner, n_outer in DEFAULT_2D_RING_GRAPH_CONFIGS)
    if dimension in {None, "3d"}:
        configs.extend(make_ring_graph_config("3d", n_inner, n_outer) for n_inner, n_outer in DEFAULT_3D_RING_GRAPH_CONFIGS)
    return configs


def parse_ring_graph_configs(configs: list[str] | None, output_dir: Path | None) -> list[RingGraphConfig]:
    if configs is None:
        return []

    parsed = []
    for config in configs:
        normalized = config.lower().replace("-", "_")
        if normalized in {"all", "all_2d", "2d", "all_3d", "3d"}:
            dimension = None if normalized == "all" else normalized.removeprefix("all_")
            discovered = discover_ring_graph_configs(output_dir, dimension)
            parsed.extend(discovered or default_ring_graph_configs(dimension))
            continue

        match = RING_GRAPH_CONFIG_RE.match(normalized)
        if match:
            parsed.append(
                make_ring_graph_config(
                    match.group("dimension"),
                    int(match.group("inner")),
                    int(match.group("outer")),
                )
            )
            continue

        parts = normalized.split(":")
        if len(parts) == 3 and parts[0] in {"2d", "3d"}:
            parsed.append(make_ring_graph_config(parts[0], int(parts[1]), int(parts[2])))
            continue

        raise ValueError(
            f"Unknown rotating-ring graph config {config!r}. "
            "Use 'all_2d', 'all_3d', 'all', '2d_3inner_4_outer', or '2d:3:4'."
        )

    deduped = []
    seen = set()
    for config in parsed:
        if config.name in seen:
            continue
        seen.add(config.name)
        deduped.append(config)
    return deduped


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


def args_for_ring_graph_config(args: argparse.Namespace, graph_config: RingGraphConfig, output_dir: Path | None = None) -> argparse.Namespace:
    config_args = argparse.Namespace(**vars(args))
    config_args.ring_n_inner = graph_config.n_inner
    config_args.ring_n_outer = graph_config.n_outer
    config_args.ring_outer_3d_rotation_deg = graph_config.outer_3d_rotation_deg
    config_args.ring_outer_3d_axis_deg = graph_config.outer_3d_axis_deg
    if output_dir is not None:
        config_args.output_dir = output_dir
    return config_args


def config_dict(args: argparse.Namespace) -> dict[str, object]:
    config = vars(args).copy()
    if isinstance(config.get("output_dir"), Path):
        config["output_dir"] = str(config["output_dir"])
    return config


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
        dataset_items = args.k
    elif args.dataset == "incompleteness":
        dataset_items = args.counterexamples
    else:
        dataset_items = ["rotating_ring"]
    total_runs = len(dataset_items) * len(args.hard_cutoffs) * len(args.interaction_layers) * len(args.seeds)
    run_index = 0

    print("Config:", flush=True, file=output)
    print(pprint.pformat(config_dict(args), sort_dicts=True), flush=True, file=output)
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
                        dist_soft_min=args.dist_soft_min,
                        dist_soft_max=dist_soft_max,
                        dist_hard_max=hard_cutoff,
                        l_max=args.l_max,
                        n_max=args.n_max,
                        stop_at_accuracy=1.0,
                        success_margin=args.success_margin,
                    )
                    results.append(train(train_args))

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
        for graph_config in graph_configs:
            graph_output_dir = args.output_dir / graph_config.name
            graph_args = args_for_ring_graph_config(args, graph_config, graph_output_dir)
            for model, l_max, n_max in configs:
                config_jobs.append((graph_args, model, l_max, n_max, graph_output_dir))
    else:
        for model, l_max, n_max in configs:
            config_jobs.append((args, model, l_max, n_max, args.output_dir))

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
