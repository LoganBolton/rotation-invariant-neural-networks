"""JSON result helpers for benchmark sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def config_dict(args: argparse.Namespace) -> dict[str, object]:
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    return config


def config_result_name(model: str, l_max: int, n_max: int) -> str:
    if model == "hipnn":
        return "l0_n1.json"
    return f"l{l_max}_n{n_max}.json"


def json_results_dir(args: argparse.Namespace) -> Path:
    if args.results_json_dir is not None:
        return args.results_json_dir
    if args.output_dir is not None:
        return args.output_dir / "json_results"
    return Path("sweep_json_results")


def sweep_dataset_items(args: argparse.Namespace) -> list[int | str]:
    if args.dataset == "k_chain":
        return list(args.k)
    if args.dataset == "incompleteness":
        return list(args.counterexamples)
    return ["rotating_ring"]


def make_run_result_record(
    args: argparse.Namespace,
    dataset_item: int | str,
    hard_cutoff: float,
    n_layers: int,
    seed: int,
    dist_soft_max: float | None,
    result: dict[str, object],
    train_time: float,
) -> dict[str, Any]:
    experiment_name = (
        f"{args.dataset}__{dataset_item}__{args.model}"
        f"__l{args.l_max}_n{args.n_max}__cutoff{hard_cutoff:g}"
        f"__layers{n_layers}__seed{seed}"
    )
    return {
        "schema_version": 1,
        "experiment_name": _sanitize_experiment_name(experiment_name),
        "status": "ok",
        "config": _run_config(args, dataset_item, hard_cutoff, n_layers, seed, dist_soft_max),
        "metrics": {
            "epoch": result["epoch"],
            "loss": result["loss"],
            "accuracy": result["accuracy"],
            "margin_accuracy": result["margin_accuracy"],
            "train_time": train_time,
        },
        "dataset": _dataset_config(args, dataset_item),
        "runtime": {},
        "logits": result["logits"],
    }


def make_sweep_result_record(args: argparse.Namespace, runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_name": _sanitize_experiment_name(f"{args.dataset}__{args.model}__l{args.l_max}_n{args.n_max}"),
        "status": "ok",
        "config": config_dict(args),
        "metrics": {
            "num_runs": len(runs),
            "mean_accuracy": _mean_metric(runs, "accuracy"),
            "mean_margin_accuracy": _mean_metric(runs, "margin_accuracy"),
            "success_rate": _success_rate(runs),
            "train_time": sum(float(run["metrics"]["train_time"]) for run in runs),
        },
        "dataset": {
            "dataset": args.dataset,
            "items": sweep_dataset_items(args),
        },
        "runtime": {},
        "runs": runs,
    }


def write_result_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_json_index(results_dir: Path) -> None:
    result_files = sorted(path for path in results_dir.glob("**/*.json") if path.name != "index.json")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in result_files]
    write_result_json(
        results_dir / "index.json",
        {
            "schema_version": 1,
            "status": "ok",
            "result_files": [str(path.relative_to(results_dir)) for path in result_files],
            "results": records,
        },
    )


def _dataset_config(args: argparse.Namespace, dataset_item: int | str) -> dict[str, Any]:
    if args.dataset == "k_chain":
        return {"dataset": args.dataset, "k": dataset_item}
    if args.dataset == "incompleteness":
        return {"dataset": args.dataset, "counterexample": dataset_item}
    return {
        "dataset": args.dataset,
        "ring_n_graphs": args.ring_n_graphs,
        "ring_seed": args.ring_seed,
        "ring_n_inner": args.ring_n_inner,
        "ring_n_outer": args.ring_n_outer,
        "ring_outer_3d_rotation_deg": args.ring_outer_3d_rotation_deg,
        "ring_outer_3d_axis_deg": args.ring_outer_3d_axis_deg,
    }


def _run_config(
    args: argparse.Namespace,
    dataset_item: int | str,
    hard_cutoff: float,
    n_layers: int,
    seed: int,
    dist_soft_max: float | None,
) -> dict[str, Any]:
    return {
        "training": {
            "epochs": args.epochs,
            "seed": seed,
            "learning_rate": args.learning_rate,
            "success_margin": args.success_margin,
            "stop_at_accuracy": 1.0,
        },
        "dataset": _dataset_config(args, dataset_item),
        "model": {
            "model": args.model,
            "neighborhood_cutoff": args.neighborhood_cutoff,
            "n_interaction_layers": n_layers,
            "n_atom_layers": args.n_atom_layers,
            "n_features": args.n_features,
            "n_sensitivities": args.n_sensitivities,
            "dist_soft_min": args.dist_soft_min,
            "dist_soft_max": dist_soft_max,
            "dist_hard_max": hard_cutoff,
            "l_max": args.l_max,
            "n_max": args.n_max,
        },
        "runtime": {},
    }


def _mean_metric(runs: list[dict[str, Any]], metric: str) -> float | None:
    values = [float(run["metrics"][metric]) for run in runs if run.get("status") == "ok"]
    return sum(values) / len(values) if values else None


def _success_rate(runs: list[dict[str, Any]]) -> float | None:
    values = [float(run["metrics"]["margin_accuracy"]) for run in runs if run.get("status") == "ok"]
    return sum(value >= 1.0 for value in values) / len(values) if values else None


def _sanitize_experiment_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)
