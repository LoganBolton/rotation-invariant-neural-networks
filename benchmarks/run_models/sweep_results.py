"""JSON result helpers for benchmark sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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
    experiment_name = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in experiment_name)
    if args.dataset == "k_chain":
        dataset = {"dataset": args.dataset, "k": dataset_item}
    elif args.dataset == "incompleteness":
        dataset = {"dataset": args.dataset, "counterexample": dataset_item}
    else:
        dataset = {
            "dataset": args.dataset,
            "ring_n_graphs": args.ring_n_graphs,
            "ring_seed": args.ring_seed,
            "ring_n_inner": args.ring_n_inner,
            "ring_n_outer": args.ring_n_outer,
            "ring_outer_3d_rotation_deg": args.ring_outer_3d_rotation_deg,
            "ring_outer_3d_axis_deg": args.ring_outer_3d_axis_deg,
        }

    return {
        "schema_version": 1,
        "experiment_name": experiment_name,
        "status": "ok",
        "config": {
            "training": {
                "epochs": args.epochs,
                "seed": seed,
                "learning_rate": args.learning_rate,
                "success_margin": args.success_margin,
                "stop_at_accuracy": 1.0,
            },
            "dataset": dataset,
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
                "group_norm": getattr(args, "hiphop_group_norm", True),
            },
            "runtime": {},
        },
        "metrics": {
            "epoch": result["epoch"],
            "loss": result["loss"],
            "accuracy": result["accuracy"],
            "margin_accuracy": result["margin_accuracy"],
            "train_time": train_time,
        },
        "dataset": dataset,
        "runtime": {},
        "logits": result["logits"],
    }


def make_sweep_result_record(args: argparse.Namespace, runs: list[dict[str, Any]]) -> dict[str, Any]:
    accuracies = [float(run["metrics"]["accuracy"]) for run in runs]
    margin_accuracies = [float(run["metrics"]["margin_accuracy"]) for run in runs]
    experiment_name = f"{args.dataset}__{args.model}__l{args.l_max}_n{args.n_max}"
    experiment_name = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in experiment_name)
    if args.dataset == "k_chain":
        items = list(args.k)
    elif args.dataset == "incompleteness":
        items = list(args.counterexamples)
    else:
        items = ["rotating_ring"]

    return {
        "schema_version": 1,
        "experiment_name": experiment_name,
        "status": "ok",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "metrics": {
            "num_runs": len(runs),
            "mean_accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
            "mean_margin_accuracy": sum(margin_accuracies) / len(margin_accuracies) if margin_accuracies else None,
            "success_rate": sum(value >= 1.0 for value in margin_accuracies) / len(margin_accuracies)
            if margin_accuracies
            else None,
            "train_time": sum(float(run["metrics"]["train_time"]) for run in runs),
        },
        "dataset": {
            "dataset": args.dataset,
            "items": items,
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
