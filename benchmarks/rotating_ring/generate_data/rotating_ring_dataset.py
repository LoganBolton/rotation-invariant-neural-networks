"""Min-distance-labeled rotating-ring graph dataset and HTML-viewer CLI.

Each graph has one center node, an inner ring, and an outer ring. Labels are
assigned from measured geometry: compute the nearest outer-node distance for
each inner-ring node, then use the single smallest of those distances for the
graph-level close/far split. In other words, the class signal is the
"minnest min", not the average of per-inner nearest distances.

This still works for equal or mixed ring sizes because it never assumes inner
and outer nodes are paired by index.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from math import gcd, pi
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

CLASS_NAMES = ("close", "far")
RING_GRAPH_CLASS_NAMES = CLASS_NAMES  # Backward-compatible public name.
DEFAULT_VISUALIZATION_DIR = Path(__file__).resolve().parent / "original_visualizations"

CENTER_ROLE = 0
INNER_ROLE = 1
OUTER_ROLE = 2
PADDING_ROLE = -1


@dataclass(frozen=True)
class RingGraphEnvironment:
    """Container for one generated graph."""

    name: str
    label: int
    Z: torch.Tensor
    R: torch.Tensor
    edge_index: torch.Tensor
    node_role: torch.Tensor
    central_atom_local_index: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return int(self.R.shape[0])

    @property
    def n_edges_directed(self) -> int:
        return int(self.edge_index.shape[1])


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _dtype(dtype: torch.dtype | None) -> torch.dtype:
    return torch.get_default_dtype() if dtype is None else dtype


def _validate_positive(**values: int | float) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")


def _validate_range(name: str, value_range: tuple[float, float]) -> None:
    low, high = value_range
    if high < low:
        raise ValueError(f"Expected {name} high >= low, got {value_range}.")


def _lerp(value_range: tuple[float, float], t: float) -> float:
    _validate_range("range", value_range)
    low, high = value_range
    return float(low + (high - low) * t)


def _sample_or_lerp(
    generator: torch.Generator,
    value_range: tuple[float, float],
    t: float,
    *,
    smooth: bool,
) -> float:
    _validate_range("range", value_range)
    if smooth:
        return _lerp(value_range, t)
    low, high = value_range
    return float(low + (high - low) * torch.rand((), generator=generator).item())


def _degrees(radians: float) -> float:
    return float(radians * 180.0 / pi)


def relative_ring_rotation_period(n_inner: int, n_outer: int) -> float:
    """Fundamental relative phase period, ``2*pi/lcm(n_inner, n_outer)``."""

    _validate_positive(n_inner=n_inner, n_outer=n_outer)
    lcm = abs(int(n_inner) * int(n_outer)) // gcd(int(n_inner), int(n_outer))
    return 2.0 * pi / float(lcm)


# -----------------------------------------------------------------------------
# Geometry and topology
# -----------------------------------------------------------------------------


def ring_positions(
    n_nodes: int,
    radius: float,
    *,
    phase_clockwise: float = 0.0,
    global_rotation: float = 0.0,
    z: float = 0.0,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Evenly spaced ``[x, y, z]`` points on a planar ring."""

    _validate_positive(n_nodes=n_nodes, radius=radius)
    ids = torch.arange(n_nodes, dtype=_dtype(dtype))
    angles = global_rotation - phase_clockwise + 2.0 * pi * ids / float(n_nodes)
    return torch.stack(
        [radius * torch.cos(angles), radius * torch.sin(angles), torch.full_like(ids, float(z))],
        dim=1,
    )


def rotate_points_about_xy_axis(
    points: torch.Tensor,
    *,
    angle: float,
    axis_angle: float = 0.0,
) -> torch.Tensor:
    """Rotate points around an axis through the origin in the xy plane."""

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape [n_points, 3], got {tuple(points.shape)}.")
    if abs(float(angle)) < 1.0e-12:
        return points.clone()

    dtype, device = points.dtype, points.device
    angle_t = torch.as_tensor(float(angle), dtype=dtype, device=device)
    axis_t = torch.as_tensor(float(axis_angle), dtype=dtype, device=device)
    axis = torch.stack(
        [torch.cos(axis_t), torch.sin(axis_t), torch.zeros((), dtype=dtype, device=device)]
    )

    cross = torch.stack(
        [
            axis[1] * points[:, 2] - axis[2] * points[:, 1],
            axis[2] * points[:, 0] - axis[0] * points[:, 2],
            axis[0] * points[:, 1] - axis[1] * points[:, 0],
        ],
        dim=1,
    )
    dot = (points * axis.view(1, 3)).sum(dim=1, keepdim=True)
    cos_a = torch.cos(angle_t)
    sin_a = torch.sin(angle_t)
    return points * cos_a + cross * sin_a + axis.view(1, 3) * dot * (1.0 - cos_a)


def _ring_slices(n_inner: int, n_outer: int) -> tuple[slice, slice]:
    _validate_positive(n_inner=n_inner, n_outer=n_outer)
    inner = slice(1, 1 + int(n_inner))
    outer = slice(1 + int(n_inner), 1 + int(n_inner) + int(n_outer))
    return inner, outer


def closest_inner_outer_distances_from_positions(
    positions: torch.Tensor,
    *,
    n_inner: int,
    n_outer: int,
) -> torch.Tensor:
    """Distance from each inner-ring node to its nearest outer-ring node."""

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected positions with shape [n_nodes, 3], got {tuple(positions.shape)}.")
    inner_slice, outer_slice = _ring_slices(n_inner, n_outer)
    required_nodes = outer_slice.stop or 0
    if positions.shape[0] < required_nodes:
        raise ValueError(f"Expected at least {required_nodes} nodes, got {positions.shape[0]}.")
    return torch.cdist(positions[inner_slice], positions[outer_slice]).min(dim=1).values


def _distance_metadata(distances: torch.Tensor) -> dict[str, Any]:
    if distances.numel() == 0:
        raise ValueError("Cannot summarize an empty distance tensor.")
    return {
        "closest_inner_outer_distances": [float(x) for x in distances.tolist()],
        "closest_inner_outer_distance_min": float(distances.min().item()),
        "closest_inner_outer_distance_mean": float(distances.mean().item()),
        "closest_inner_outer_distance_max": float(distances.max().item()),
    }


def bidirectional_edge_index(edges: Iterable[tuple[int, int]]) -> torch.Tensor:
    """Convert undirected edge pairs to sorted directed ``edge_index``."""

    directed = {
        edge
        for src, dst in edges
        for edge in ((int(src), int(dst)), (int(dst), int(src)))
        if int(src) != int(dst)
    }
    if not directed:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(sorted(directed), dtype=torch.long).t().contiguous()


def undirected_edge_pairs(edge_index: torch.Tensor) -> list[tuple[int, int]]:
    """Return unique undirected edge pairs from directed ``edge_index``."""

    return sorted({tuple(sorted(map(int, edge))) for edge in edge_index.t().tolist() if edge[0] != edge[1]})


def ring_graph_edge_index(
    *,
    n_inner: int,
    n_outer: int,
    add_inner_ring_edges: bool = False,
    add_outer_ring_edges: bool = False,
    add_center_outer_edges: bool = False,
) -> torch.Tensor:
    """Center-to-ring spokes plus optional inner/outer ring-cycle edges."""

    _validate_positive(n_inner=n_inner, n_outer=n_outer)
    inner_nodes = list(range(1, 1 + n_inner))
    outer_nodes = list(range(1 + n_inner, 1 + n_inner + n_outer))
    edges = [(0, node) for node in inner_nodes + outer_nodes]

    # Kept only for CLI compatibility; center-to-outer spokes are already default.
    _ = add_center_outer_edges

    if add_inner_ring_edges:
        edges += [(inner_nodes[i], inner_nodes[(i + 1) % n_inner]) for i in range(n_inner)]
    if add_outer_ring_edges:
        edges += [(outer_nodes[i], outer_nodes[(i + 1) % n_outer]) for i in range(n_outer)]
    return bidirectional_edge_index(edges)


# -----------------------------------------------------------------------------
# Dataset generation
# -----------------------------------------------------------------------------


def create_ring_graph_environment(
    *,
    label: int,
    inner_radius: float,
    outer_radius: float,
    outer_rotation_clockwise: float,
    outer_3d_rotation: float = 0.0,
    outer_3d_axis_angle: float = 0.0,
    global_rotation: float = 0.0,
    n_inner: int = 8,
    n_outer: int = 8,
    class_phase_offset_fraction: float = 0.0,
    name: str | None = None,
    add_inner_ring_edges: bool = False,
    add_outer_ring_edges: bool = False,
    add_center_outer_edges: bool = False,
    dtype: torch.dtype | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RingGraphEnvironment:
    """Create one graph and record nearest-inner-to-outer distance diagnostics."""

    if label not in (0, 1):
        raise ValueError(f"label must be 0 or 1, got {label}.")
    if outer_radius <= inner_radius:
        raise ValueError(f"outer_radius must be larger than inner_radius, got {outer_radius} <= {inner_radius}.")

    dtype = _dtype(dtype)
    phase_offset = label * class_phase_offset_fraction * (2.0 * pi / float(n_inner))
    outer_phase = outer_rotation_clockwise + phase_offset

    inner = ring_positions(n_inner, inner_radius, global_rotation=global_rotation, dtype=dtype)
    outer = ring_positions(
        n_outer,
        outer_radius,
        phase_clockwise=outer_phase,
        global_rotation=global_rotation,
        dtype=dtype,
    )
    outer = rotate_points_about_xy_axis(
        outer,
        angle=outer_3d_rotation,
        axis_angle=global_rotation + outer_3d_axis_angle,
    )
    positions = torch.cat([torch.zeros((1, 3), dtype=dtype), inner, outer], dim=0)
    distances = closest_inner_outer_distances_from_positions(positions, n_inner=n_inner, n_outer=n_outer)

    sample_metadata: dict[str, Any] = {
        "class_name": CLASS_NAMES[label],
        "n_inner": int(n_inner),
        "n_outer": int(n_outer),
        "inner_radius": float(inner_radius),
        "outer_radius": float(outer_radius),
        "outer_gap": float(outer_radius - inner_radius),
        "outer_rotation_clockwise": float(outer_rotation_clockwise),
        "outer_3d_rotation": float(outer_3d_rotation),
        "outer_3d_axis_angle": float(outer_3d_axis_angle),
        "outer_3d_rotation_deg": _degrees(outer_3d_rotation),
        "outer_3d_axis_deg": _degrees(outer_3d_axis_angle),
        "global_rotation": float(global_rotation),
        "class_phase_offset_clockwise": float(phase_offset),
        "outer_phase_clockwise": float(outer_phase),
        "class_phase_offset_fraction": float(class_phase_offset_fraction),
        "add_inner_ring_edges": bool(add_inner_ring_edges),
        "add_outer_ring_edges": bool(add_outer_ring_edges),
        "add_center_outer_edges": bool(add_center_outer_edges),
        **_distance_metadata(distances),
    }
    if metadata:
        sample_metadata.update(dict(metadata))

    n_nodes = positions.shape[0]
    return RingGraphEnvironment(
        name=name or f"ring_graph_{CLASS_NAMES[label]}",
        label=label,
        Z=torch.ones(n_nodes, dtype=torch.long),
        R=positions,
        edge_index=ring_graph_edge_index(
            n_inner=n_inner,
            n_outer=n_outer,
            add_inner_ring_edges=add_inner_ring_edges,
            add_outer_ring_edges=add_outer_ring_edges,
            add_center_outer_edges=add_center_outer_edges,
        ),
        node_role=torch.tensor([CENTER_ROLE] + [INNER_ROLE] * n_inner + [OUTER_ROLE] * n_outer),
        metadata=sample_metadata,
    )


def _generation_ranges_valid(ranges: Mapping[str, tuple[float, float]]) -> None:
    for name, value_range in ranges.items():
        _validate_range(name, value_range)


def create_rotating_ring_dataset(
    *,
    n_graphs: int = 500,
    seed: int = 0,
    n_inner: int = 8,
    n_outer: int = 8,
    inner_radius_range: tuple[float, float] = (1.0, 1.0),
    outer_gap_range: tuple[float, float] = (1.2, 1.2),
    outer_rotation_fraction_range: tuple[float, float] = (0.0, 0.5),
    outer_3d_rotation_range: tuple[float, float] = (0.0, 0.0),
    outer_3d_axis_angle: float = 0.0,
    global_rotation_fraction_range: tuple[float, float] = (0.0, 0.0),
    distance_split_statistic: str = "min",
    smooth_order: bool = True,
    shuffle: bool = False,
    add_inner_ring_edges: bool = False,
    add_outer_ring_edges: bool = False,
    add_center_outer_edges: bool = False,
    dtype: torch.dtype | None = None,
) -> list[RingGraphEnvironment]:
    """Generate graphs, then label halves by each graph's minimum nearest distance.

    ``distance_split_statistic`` is kept as an API-compatible keyword, but this
    variant intentionally only supports ``"min"`` so the split is based on the
    single smallest inner-to-outer nearest distance in each graph.
    """

    if n_graphs <= 1:
        raise ValueError(f"n_graphs must be at least 2 for a balanced split, got {n_graphs}.")
    _validate_positive(n_inner=n_inner, n_outer=n_outer)
    if distance_split_statistic != "min":
        raise ValueError(
            "This min-distance dataset variant only supports "
            "distance_split_statistic='min'."
        )

    ranges = {
        "inner_radius_range": inner_radius_range,
        "outer_gap_range": outer_gap_range,
        "outer_rotation_fraction_range": outer_rotation_fraction_range,
        "outer_3d_rotation_range": outer_3d_rotation_range,
        "global_rotation_fraction_range": global_rotation_fraction_range,
    }
    _generation_ranges_valid(ranges)

    generator = torch.Generator().manual_seed(int(seed))
    relative_period = relative_ring_rotation_period(n_inner, n_outer)
    inner_step = 2.0 * pi / float(n_inner)
    candidates: list[RingGraphEnvironment] = []

    for index in range(n_graphs):
        t = 0.0 if n_graphs <= 1 else index / float(n_graphs - 1)
        inner_radius = _sample_or_lerp(generator, inner_radius_range, t, smooth=smooth_order)
        outer_gap = _sample_or_lerp(generator, outer_gap_range, t, smooth=smooth_order)
        rotation_fraction = _sample_or_lerp(generator, outer_rotation_fraction_range, t, smooth=smooth_order)
        tilt = _sample_or_lerp(generator, outer_3d_rotation_range, t, smooth=smooth_order)
        global_fraction = _sample_or_lerp(generator, global_rotation_fraction_range, t, smooth=smooth_order)

        candidates.append(
            create_ring_graph_environment(
                label=0,
                inner_radius=inner_radius,
                outer_radius=inner_radius + outer_gap,
                outer_rotation_clockwise=rotation_fraction * relative_period,
                outer_3d_rotation=tilt,
                outer_3d_axis_angle=outer_3d_axis_angle,
                global_rotation=global_fraction * inner_step,
                n_inner=n_inner,
                n_outer=n_outer,
                name=f"rotating_ring_unlabeled_{index:04d}",
                add_inner_ring_edges=add_inner_ring_edges,
                add_outer_ring_edges=add_outer_ring_edges,
                add_center_outer_edges=add_center_outer_edges,
                dtype=dtype,
                metadata={
                    "seed": int(seed),
                    "candidate_index": int(index),
                    "generation_index": int(index),
                    "variation_t": float(t),
                    "smooth_order": bool(smooth_order),
                    "outer_rotation_fraction": float(rotation_fraction),
                    "outer_rotation_fraction_basis": "relative_phase_period",
                    "relative_rotation_period": float(relative_period),
                    "relative_rotation_period_deg": _degrees(relative_period),
                    "outer_3d_rotation_range": tuple(map(float, outer_3d_rotation_range)),
                    "outer_3d_rotation_deg_range": tuple(_degrees(x) for x in outer_3d_rotation_range),
                    "outer_3d_axis_angle": float(outer_3d_axis_angle),
                    "outer_3d_axis_deg": _degrees(outer_3d_axis_angle),
                    "global_rotation_fraction": float(global_fraction),
                    "class_distance_mode": "minimum_nearest_distance_split",
                    "distance_split_statistic": "min",
                    "distance_histogram_value_name": "closest_inner_outer_distance_min",
                    "inner_radius_range": tuple(map(float, inner_radius_range)),
                    "outer_gap_range": tuple(map(float, outer_gap_range)),
                    "outer_rotation_fraction_range": tuple(map(float, outer_rotation_fraction_range)),
                    "global_rotation_fraction_range": tuple(map(float, global_rotation_fraction_range)),
                },
            )
        )

    key = f"closest_inner_outer_distance_{distance_split_statistic}"
    sorted_candidates = sorted(candidates, key=lambda env: (float(env.metadata[key]), env.metadata["candidate_index"]))
    per_class = n_graphs // 2
    class_groups = (sorted_candidates[:per_class], sorted_candidates[-per_class:])
    thresholds = (float(class_groups[0][-1].metadata[key]), float(class_groups[1][0].metadata[key]))

    relabeled: dict[int, RingGraphEnvironment] = {}
    for label, group in enumerate(class_groups):
        for class_index, env in enumerate(group):
            candidate_index = int(env.metadata["candidate_index"])
            metadata = dict(env.metadata)
            metadata.update(
                class_name=CLASS_NAMES[label],
                class_index=int(class_index),
                balanced_class_count=int(per_class),
                requested_n_graphs=int(n_graphs),
                returned_n_graphs=int(2 * per_class),
                dropped_for_balance=int(n_graphs - 2 * per_class),
                distance_split_rank=int(sorted_candidates.index(env)),
                distance_split_threshold_low=thresholds[0],
                distance_split_threshold_high=thresholds[1],
                active_outer_gap_range=tuple(map(float, outer_gap_range)),
            )
            relabeled[candidate_index] = replace(
                env,
                label=label,
                name=f"rotating_ring_{CLASS_NAMES[label]}_{class_index:04d}",
                metadata=metadata,
            )

    environments = [relabeled[i] for i in sorted(relabeled)]
    if shuffle:
        order = torch.randperm(len(environments), generator=generator).tolist()
        environments = [environments[i] for i in order]
    return environments


# -----------------------------------------------------------------------------
# Conversion helpers
# -----------------------------------------------------------------------------


def as_padded_ring_arrays(
    environments: list[RingGraphEnvironment],
    *,
    center_on_central_node: bool = True,
) -> dict[str, torch.Tensor]:
    """Stack variable-size graphs into padded tensors plus compressed edges."""

    if not environments:
        raise ValueError("Cannot stack an empty environment list.")

    n_graphs = len(environments)
    max_nodes = max(env.n_nodes for env in environments)
    max_edges = max(env.n_edges_directed for env in environments)
    dtype = environments[0].R.dtype

    arrays = {
        "Z": torch.zeros((n_graphs, max_nodes), dtype=torch.long),
        "R": torch.zeros((n_graphs, max_nodes, 3), dtype=dtype),
        "T": torch.empty((n_graphs, 1), dtype=torch.get_default_dtype()),
        "edge_indices": torch.full((n_graphs, 2, max_edges), -1, dtype=torch.long),
        "node_role": torch.full((n_graphs, max_nodes), PADDING_ROLE, dtype=torch.long),
        "node_mask": torch.zeros((n_graphs, max_nodes), dtype=torch.get_default_dtype()),
        "node_counts": torch.empty((n_graphs,), dtype=torch.long),
    }
    graph_index_parts: list[torch.Tensor] = []
    edge_index_parts: list[torch.Tensor] = []
    atom_offset = 0

    for graph_index, env in enumerate(environments):
        n = env.n_nodes
        positions = env.R.clone()
        if center_on_central_node:
            positions = positions - positions[env.central_atom_local_index : env.central_atom_local_index + 1]

        arrays["Z"][graph_index, :n] = env.Z
        arrays["R"][graph_index, :n] = positions
        arrays["T"][graph_index, 0] = float(env.label)
        arrays["edge_indices"][graph_index, :, : env.n_edges_directed] = env.edge_index
        arrays["node_role"][graph_index, :n] = env.node_role
        arrays["node_mask"][graph_index, :n] = 1.0
        arrays["node_counts"][graph_index] = n

        edge_index_parts.append(env.edge_index + atom_offset)
        graph_index_parts.append(torch.full((n,), graph_index, dtype=torch.long))
        atom_offset += n

    arrays["edge_index"] = torch.cat(edge_index_parts, dim=1)
    arrays["graph_index"] = torch.cat(graph_index_parts, dim=0)
    return arrays


def as_hippynn_arrays(
    environments: list[RingGraphEnvironment],
    *,
    center: bool = True,
) -> dict[str, torch.Tensor]:
    return as_padded_ring_arrays(environments, center_on_central_node=center)


def as_pyg_data_list(environments: list[RingGraphEnvironment]) -> list[Any]:
    """Convert to PyTorch Geometric ``Data`` objects."""

    try:
        from torch_geometric.data import Data
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install torch_geometric to use as_pyg_data_list().") from exc

    return [
        Data(
            z=env.Z.clone(),
            pos=env.R.clone(),
            edge_index=env.edge_index.clone(),
            y=torch.tensor([env.label], dtype=torch.long),
            node_role=env.node_role.clone(),
            name=env.name,
            metadata=dict(env.metadata),
        )
        for env in environments
    ]


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a distance-labeled rotating-ring HTML viewer.")
    parser.add_argument("--n-graphs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-inner", type=int, default=8)
    parser.add_argument("--n-outer", type=int, default=8)
    parser.add_argument("--html", type=Path, default=DEFAULT_VISUALIZATION_DIR / "rotating_ring_viewer.html")
    parser.add_argument("--viewer-max-graphs", type=int, default=500)
    parser.add_argument("--inner-radius-min", type=float, default=1.0)
    parser.add_argument("--inner-radius-max", type=float, default=1.0)
    parser.add_argument("--outer-gap-min", type=float, default=1.2)
    parser.add_argument("--outer-gap-max", type=float, default=1.2)
    parser.add_argument("--distance-split-statistic", choices=("min",), default="min")
    parser.add_argument("--outer-rotation-frac-min", type=float, default=0.0)
    parser.add_argument("--outer-rotation-frac-max", type=float, default=0.5)
    parser.add_argument("--outer-3d-rotation-deg", type=float, default=None)
    parser.add_argument("--outer-3d-rotation-deg-min", type=float, default=0.0)
    parser.add_argument("--outer-3d-rotation-deg-max", type=float, default=0.0)
    parser.add_argument("--outer-3d-axis-deg", type=float, default=0.0)
    parser.add_argument("--global-rotation-frac-min", type=float, default=0.0)
    parser.add_argument("--global-rotation-frac-max", type=float, default=0.0)
    parser.add_argument("--random-parameters", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--add-inner-ring-edges", action="store_true")
    parser.add_argument("--add-outer-ring-edges", action="store_true")
    parser.add_argument("--add-center-outer-edges", action="store_true")
    return parser.parse_args()


def _tilt_range_from_args(args: argparse.Namespace) -> tuple[float, float]:
    max_deg = args.outer_3d_rotation_deg if args.outer_3d_rotation_deg is not None else args.outer_3d_rotation_deg_max
    return (args.outer_3d_rotation_deg_min * pi / 180.0, max_deg * pi / 180.0)


def main() -> None:
    args = _parse_args()
    envs = create_rotating_ring_dataset(
        n_graphs=args.n_graphs,
        seed=args.seed,
        n_inner=args.n_inner,
        n_outer=args.n_outer,
        inner_radius_range=(args.inner_radius_min, args.inner_radius_max),
        outer_gap_range=(args.outer_gap_min, args.outer_gap_max),
        outer_rotation_fraction_range=(args.outer_rotation_frac_min, args.outer_rotation_frac_max),
        outer_3d_rotation_range=_tilt_range_from_args(args),
        outer_3d_axis_angle=args.outer_3d_axis_deg * pi / 180.0,
        global_rotation_fraction_range=(args.global_rotation_frac_min, args.global_rotation_frac_max),
        distance_split_statistic=args.distance_split_statistic,
        smooth_order=not args.random_parameters,
        shuffle=args.shuffle,
        add_inner_ring_edges=args.add_inner_ring_edges,
        add_outer_ring_edges=args.add_outer_ring_edges,
        add_center_outer_edges=args.add_center_outer_edges,
    )

    try:
        from .rotating_ring_viewer import VIEWER_VERSION, write_ring_graph_viewer
    except ImportError:
        from rotating_ring_viewer import VIEWER_VERSION, write_ring_graph_viewer

    write_ring_graph_viewer(envs, args.html, max_graphs=args.viewer_max_graphs)
    print(f"saved viewer for {len(envs)} graphs to {args.html} using viewer version {VIEWER_VERSION}")


if __name__ == "__main__":
    main()
