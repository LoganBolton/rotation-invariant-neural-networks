"""Export rotating-ring graph datasets as colleague-friendly NPZ files.

The export layout is:

    output_dir/
      dataset_metadata.json
      manifest.csv
      inner1_outer1/
        config_metadata.json
        viewer.html
        graphs/
          graph_0000_label0_close.npz
          ...

Each per-graph NPZ contains plain NumPy arrays:

    positions       float array, shape [n_nodes, 3]
    atomic_numbers  int array, shape [n_nodes]
    edge_index      int array, shape [2, n_directed_edges]
    node_role       int array, shape [n_nodes], 0=center, 1=inner, 2=outer
    label           int scalar, 0=close, 1=far

The NPZ also includes ``metadata_json`` as a string for convenience. The
top-level manifest is usually the easiest way to scan labels and geometry.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

BENCHMARKS_ROOT = Path(__file__).resolve().parents[2]
if str(BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_ROOT))

from rotating_ring.generate_data.rotating_ring_dataset import (  # noqa: E402
    CENTER_ROLE,
    CLASS_NAMES,
    INNER_ROLE,
    OUTER_ROLE,
    create_rotating_ring_dataset,
)
from rotating_ring.generate_data.rotating_ring_viewer import write_ring_graph_viewer  # noqa: E402


DEFAULT_2D_RING_GRAPH_CONFIGS = (
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
    (4, 1),
    (4, 2),
    (4, 3),
    (4, 4),
)
MANIFEST_FIELDS = [
    "config",
    "graph_index",
    "graph_file",
    "viewer_file",
    "name",
    "label",
    "class_name",
    "n_inner",
    "n_outer",
    "n_nodes",
    "n_edges_directed",
    "inner_radius",
    "outer_radius",
    "outer_gap",
    "outer_rotation_fraction",
    "outer_phase_clockwise",
    "outer_3d_rotation_deg",
    "outer_3d_axis_deg",
    "closest_inner_outer_distance_min",
    "closest_inner_outer_distance_mean",
    "closest_inner_outer_distance_max",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/rotating_ring/export_data/default_2d_ring_graphs_seed0_100"),
    )
    parser.add_argument("--n-graphs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--viewer-max-graphs", type=int, default=100)
    parser.add_argument("--skip-viewers", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def graph_filename(graph_index: int, label: int) -> str:
    return f"graph_{graph_index:04d}_label{label}_{CLASS_NAMES[label]}.npz"


def config_name(config: tuple[int, int]) -> str:
    return f"{config[0]}_inner_{config[1]}_outer"


def export_graph_npz(path: Path, env, *, config: tuple[int, int], graph_index: int) -> None:
    metadata = {
        **dict(env.metadata),
        "config_inner": int(config[0]),
        "config_outer": int(config[1]),
        "graph_index_in_config": int(graph_index),
        "role_names": {
            str(CENTER_ROLE): "center",
            str(INNER_ROLE): "inner",
            str(OUTER_ROLE): "outer",
        },
    }
    np.savez_compressed(
        path,
        positions=env.R.detach().cpu().numpy(),
        atomic_numbers=env.Z.detach().cpu().numpy(),
        edge_index=env.edge_index.detach().cpu().numpy(),
        node_role=env.node_role.detach().cpu().numpy(),
        label=np.asarray(env.label, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(json_ready(metadata), sort_keys=True)),
    )


def manifest_row(
    *,
    output_dir: Path,
    config_dir: Path,
    graph_path: Path,
    viewer_path: Path | None,
    env,
    config: tuple[int, int],
    graph_index: int,
) -> dict[str, Any]:
    metadata = dict(env.metadata)
    return {
        "config": config_name(config),
        "graph_index": graph_index,
        "graph_file": graph_path.relative_to(output_dir),
        "viewer_file": "" if viewer_path is None else viewer_path.relative_to(output_dir),
        "name": env.name,
        "label": int(env.label),
        "class_name": metadata.get("class_name", CLASS_NAMES[int(env.label)]),
        "n_inner": config[0],
        "n_outer": config[1],
        "n_nodes": env.n_nodes,
        "n_edges_directed": env.n_edges_directed,
        "inner_radius": metadata.get("inner_radius", ""),
        "outer_radius": metadata.get("outer_radius", ""),
        "outer_gap": metadata.get("outer_gap", ""),
        "outer_rotation_fraction": metadata.get("outer_rotation_fraction", ""),
        "outer_phase_clockwise": metadata.get("outer_phase_clockwise", ""),
        "outer_3d_rotation_deg": metadata.get("outer_3d_rotation_deg", ""),
        "outer_3d_axis_deg": metadata.get("outer_3d_axis_deg", ""),
        "closest_inner_outer_distance_min": metadata.get("closest_inner_outer_distance_min", ""),
        "closest_inner_outer_distance_mean": metadata.get("closest_inner_outer_distance_mean", ""),
        "closest_inner_outer_distance_max": metadata.get("closest_inner_outer_distance_max", ""),
    }


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in MANIFEST_FIELDS})


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    for config in DEFAULT_2D_RING_GRAPH_CONFIGS:
        inner, outer = config
        config_dir = args.output_dir / config_name(config)
        graphs_dir = config_dir / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)

        environments = create_rotating_ring_dataset(
            n_graphs=args.n_graphs,
            seed=args.seed,
            n_inner=inner,
            n_outer=outer,
        )
        viewer_path = None
        if not args.skip_viewers:
            viewer_path = config_dir / "viewer.html"
            write_ring_graph_viewer(environments, viewer_path, max_graphs=args.viewer_max_graphs)

        write_json(
            config_dir / "config_metadata.json",
            {
                "config": {"n_inner": inner, "n_outer": outer},
                "n_graphs_requested": args.n_graphs,
                "n_graphs_exported": len(environments),
                "seed": args.seed,
                "class_names": {0: CLASS_NAMES[0], 1: CLASS_NAMES[1]},
                "role_names": {CENTER_ROLE: "center", INNER_ROLE: "inner", OUTER_ROLE: "outer"},
            },
        )

        for graph_index, env in enumerate(environments):
            graph_path = graphs_dir / graph_filename(graph_index, int(env.label))
            export_graph_npz(graph_path, env, config=config, graph_index=graph_index)
            manifest_rows.append(
                manifest_row(
                    output_dir=args.output_dir,
                    config_dir=config_dir,
                    graph_path=graph_path,
                    viewer_path=viewer_path,
                    env=env,
                    config=config,
                    graph_index=graph_index,
                )
            )

        print(f"exported {len(environments)} graphs for inner={inner}, outer={outer} to {config_dir}")

    write_manifest(args.output_dir / "manifest.csv", manifest_rows)
    write_json(
        args.output_dir / "dataset_metadata.json",
        {
            "description": "Default 2D rotating-ring graph grid exported as per-graph NPZ files.",
            "n_graphs_per_config_requested": args.n_graphs,
            "n_graphs_total_exported": len(manifest_rows),
            "seed": args.seed,
            "configs": [{"n_inner": inner, "n_outer": outer} for inner, outer in DEFAULT_2D_RING_GRAPH_CONFIGS],
            "files": {
                "manifest": "manifest.csv",
                "per_config_metadata": "N_inner_N_outer/config_metadata.json",
                "per_config_viewer": "N_inner_N_outer/viewer.html",
                "per_graph_npz": "N_inner_N_outer/graphs/graph_####_label#_class.npz",
            },
            "npz_arrays": {
                "positions": "[n_nodes, 3] float coordinates",
                "atomic_numbers": "[n_nodes] integer species values",
                "edge_index": "[2, n_directed_edges] directed source/destination indices",
                "node_role": "[n_nodes] 0=center, 1=inner, 2=outer",
                "label": "scalar int, 0=close, 1=far",
                "metadata_json": "JSON string with graph geometry and distance metadata",
            },
            "class_names": {0: CLASS_NAMES[0], 1: CLASS_NAMES[1]},
            "role_names": {CENTER_ROLE: "center", INNER_ROLE: "inner", OUTER_ROLE: "outer"},
        },
    )
    print(f"wrote manifest with {len(manifest_rows)} graphs to {args.output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
