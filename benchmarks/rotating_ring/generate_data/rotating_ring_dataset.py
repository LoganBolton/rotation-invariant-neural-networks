"""Rotating-ring graph dataset helpers and CLI.

This file creates a two-class synthetic graph dataset similar to the sketch in the
prompt: one central node, one inner ring, and one outer ring in 3D coordinates.
By default both rings are planar (z=0), but the outer ring can optionally be
tilted out of the xy plane while the inner ring remains planar.

Default class definitions
-------------------------
All samples use the same inner-radius range and the same outer-gap / outer-radius
range. Labels are assigned only after geometry is generated: the graphs are sorted
by their realized closest inner-to-outer distance statistic, then the lower half is
label 0 ("close") and the upper half is label 1 ("far"). If the requested number
of graphs is odd, the median-distance graph is dropped so the final dataset is
balanced by rounding down.

The class label is therefore encoded by the closest-distance distribution that
results from outer-ring rotation / optional 3D tilt, not by class-specific radii,
absolute orientation, label-specific angular phase, node count, or atom/species
type.

The topology is undirected and, by default, contains center-to-inner and
center-to-outer spoke edges. Ring-cycle edges can be enabled with flags if desired.

3D:
python generate_data/rotating_ring_dataset.py \
  --n-graphs 500 \
  --seed 7 \
  --outer-3d-rotation-deg 70 \
  --outer-3d-axis-deg 35 \
  --html generate_data/original_visualizations/rotating_ring_viewer_outer3d.html

2D:
python generate_data/rotating_ring_dataset.py \
  --n-graphs 500 \
  --seed 7 \
  --html generate_data/original_visualizations/rotating_ring_viewer.html
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from math import pi
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


RING_GRAPH_CLASS_NAMES = ("close", "far")
DEFAULT_VISUALIZATION_DIR = Path(__file__).resolve().parent / "original_visualizations"

# Node-role values used only for bookkeeping/viewing. Z is left as all ones by
# default so that atom/species type does not leak the class label.
CENTER_ROLE = 0
INNER_ROLE = 1
OUTER_ROLE = 2
PADDING_ROLE = -1


@dataclass(frozen=True)
class RingGraphEnvironment:
    """A single rotating-ring graph.

    Attributes
    ----------
    name:
        Human-readable sample name.
    label:
        Integer class label. 0 = lower half of closest-distance statistic, 1 = upper half.
    Z:
        Node species/types with shape [n_nodes]. Defaults to all ones.
    R:
        3D node positions with shape [n_nodes, 3]. The default generator places
        all nodes in the xy plane, so z = 0. When outer 3D tilt is enabled,
        only the outer-ring nodes move out of the xy plane.
    edge_index:
        Directed edge list with shape [2, n_directed_edges]. Edges are stored in
        both directions for an undirected graph.
    node_role:
        Role IDs with shape [n_nodes], useful for coloring the viewer.
    central_atom_local_index:
        Index of the center node. Always 0 for the generated graphs.
    metadata:
        Per-graph scalar values such as radii and rotations.
    """

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
# Geometry and graph construction
# -----------------------------------------------------------------------------


def _as_dtype(dtype: torch.dtype | None = None) -> torch.dtype:
    return torch.get_default_dtype() if dtype is None else dtype


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    if high < low:
        raise ValueError(f"Expected high >= low, got low={low} and high={high}.")
    return float(low + (high - low) * torch.rand((), generator=generator).item())


def ring_positions(
    n_nodes: int,
    radius: float,
    *,
    phase_clockwise: float = 0.0,
    global_rotation: float = 0.0,
    z: float = 0.0,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return evenly spaced points on a planar ring as [n_nodes, 3].

    `phase_clockwise` is positive for a clockwise visual rotation in the xy plane
    when y is plotted upward. Mathematically, that means it is subtracted from
    the usual counter-clockwise angle.
    """

    if n_nodes <= 0:
        raise ValueError(f"n_nodes must be positive, got {n_nodes}.")
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}.")

    dtype = _as_dtype(dtype)
    ids = torch.arange(n_nodes, dtype=dtype)
    angles = global_rotation - phase_clockwise + 2.0 * pi * ids / float(n_nodes)
    x = radius * torch.cos(angles)
    y = radius * torch.sin(angles)
    z_values = torch.full_like(x, float(z))
    return torch.stack([x, y, z_values], dim=1)




def rotate_points_about_xy_axis(
    points: torch.Tensor,
    *,
    angle: float,
    axis_angle: float = 0.0,
) -> torch.Tensor:
    """Rotate points around an axis that lies in the xy plane.

    Parameters
    ----------
    points:
        Tensor with shape [n_points, 3]. Each row is rotated around the axis
        passing through the origin.
    angle:
        Right-hand-rule rotation angle in radians. A value of 0 keeps the ring
        planar, which is the default 2D behavior.
    axis_angle:
        Direction of the rotation axis inside the xy plane, in radians. The
        default 0 rotates around the positive x-axis. Values are measured
        counter-clockwise from +x toward +y.

    Rotating a planar outer ring this way keeps every outer node at the same
    distance from the center, so the nodes move on the surface of a sphere with
    radius equal to the outer-ring radius.
    """

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape [n_points, 3], got {tuple(points.shape)}.")

    angle_value = float(angle)
    if abs(angle_value) < 1.0e-12:
        return points.clone()

    dtype = points.dtype
    device = points.device
    angle_tensor = torch.as_tensor(angle_value, dtype=dtype, device=device)
    axis_angle_tensor = torch.as_tensor(float(axis_angle), dtype=dtype, device=device)
    k = torch.stack(
        [
            torch.cos(axis_angle_tensor),
            torch.sin(axis_angle_tensor),
            torch.zeros((), dtype=dtype, device=device),
        ]
    )

    cos_angle = torch.cos(angle_tensor)
    sin_angle = torch.sin(angle_tensor)

    # Rodrigues rotation formula: v_rot = v cos(a) + (k x v) sin(a) + k(k dot v)(1 - cos(a)).
    k_cross_v = torch.stack(
        [
            k[1] * points[:, 2] - k[2] * points[:, 1],
            k[2] * points[:, 0] - k[0] * points[:, 2],
            k[0] * points[:, 1] - k[1] * points[:, 0],
        ],
        dim=1,
    )
    k_dot_v = (points * k.view(1, 3)).sum(dim=1, keepdim=True)
    return points * cos_angle + k_cross_v * sin_angle + k.view(1, 3) * k_dot_v * (1.0 - cos_angle)


def closest_inner_outer_distances_from_positions(
    positions: torch.Tensor,
    *,
    n_inner: int,
    n_outer: int,
) -> torch.Tensor:
    """Return each inner node's distance to its closest outer-ring node.

    The returned tensor has shape [n_inner]. It is computed directly from the
    final 3D coordinates, so outer-ring rotation, global rotation, and optional
    3D tilt are all included in the distance calculation.
    """

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected positions with shape [n_nodes, 3], got {tuple(positions.shape)}.")
    if n_inner <= 0:
        raise ValueError(f"n_inner must be positive, got {n_inner}.")
    if n_outer <= 0:
        raise ValueError(f"n_outer must be positive, got {n_outer}.")

    inner_start = 1
    outer_start = 1 + int(n_inner)
    outer_end = outer_start + int(n_outer)
    if positions.shape[0] < outer_end:
        raise ValueError(
            f"Expected at least {outer_end} nodes for n_inner={n_inner}, n_outer={n_outer}; "
            f"got {positions.shape[0]}."
        )

    inner = positions[inner_start:outer_start]
    outer = positions[outer_start:outer_end]
    pairwise_distances = torch.cdist(inner, outer, p=2.0)
    return pairwise_distances.min(dim=1).values


def _distance_summary(values: torch.Tensor) -> dict[str, float]:
    """Return min/mean/max summary statistics for a distance tensor."""

    if values.numel() == 0:
        raise ValueError("Cannot summarize an empty distance tensor.")
    return {
        "min": float(values.min().item()),
        "mean": float(values.mean().item()),
        "max": float(values.max().item()),
    }


def bidirectional_edge_index(edges: Iterable[tuple[int, int]]) -> torch.Tensor:
    """Convert an iterable of undirected edges to a directed edge_index tensor."""

    directed: set[tuple[int, int]] = set()
    for src, dst in edges:
        src = int(src)
        dst = int(dst)
        if src == dst:
            continue
        directed.add((src, dst))
        directed.add((dst, src))

    if not directed:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(sorted(directed), dtype=torch.long).t().contiguous()


def undirected_edge_pairs(edge_index: torch.Tensor) -> list[tuple[int, int]]:
    """Return unique undirected edges from a directed edge_index tensor."""

    pairs: set[tuple[int, int]] = set()
    for src, dst in edge_index.t().tolist():
        if src == dst:
            continue
        a, b = sorted((int(src), int(dst)))
        pairs.add((a, b))
    return sorted(pairs)


def ring_graph_edge_index(
    *,
    n_inner: int,
    n_outer: int,
    add_inner_ring_edges: bool = False,
    add_outer_ring_edges: bool = False,
    add_center_outer_edges: bool = False,
) -> torch.Tensor:
    """Build the undirected topology for a center-spoke ring graph.

    Node indexing convention:
        0                         center
        1 .. n_inner              inner ring
        1 + n_inner .. end        outer ring

    By default every non-center node connects directly to the center. There are
    no inner-to-outer spoke edges, so changing outer-ring rotation or tilt keeps
    exactly the same node positions while changing only where the center spokes
    point.

    The optional ring-cycle flags still add edges within the inner and/or outer
    rings when explicitly requested.
    """

    if n_inner <= 0:
        raise ValueError(f"n_inner must be positive, got {n_inner}.")
    if n_outer <= 0:
        raise ValueError(f"n_outer must be positive, got {n_outer}.")

    center = 0
    inner_start = 1
    outer_start = 1 + n_inner
    edges: list[tuple[int, int]] = []

    # Spokes from the center to every inner-ring node.
    for i in range(n_inner):
        edges.append((center, inner_start + i))

    # Spokes from the center directly to every outer-ring node.
    for j in range(n_outer):
        edges.append((center, outer_start + j))

    # Kept for backward CLI compatibility. Center-to-outer edges are now the
    # default topology, so this flag does not add any additional edges.
    if add_center_outer_edges:
        pass

    if add_inner_ring_edges:
        for i in range(n_inner):
            edges.append((inner_start + i, inner_start + ((i + 1) % n_inner)))

    if add_outer_ring_edges:
        for j in range(n_outer):
            edges.append((outer_start + j, outer_start + ((j + 1) % n_outer)))

    return bidirectional_edge_index(edges)


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
    """Create one ring graph sample.

    Parameters
    ----------
    label:
        0 for close inner-to-outer distances, 1 for far inner-to-outer distances.
    inner_radius, outer_radius:
        Ring radii. outer_radius must be larger than inner_radius.
    outer_rotation_clockwise:
        Per-sample clockwise phase of the outer ring, in radians. This changes
        continuously across generated samples.
    outer_3d_rotation:
        Out-of-plane tilt angle for the outer ring, in radians. The default 0
        keeps the current 2D behavior. A nonzero value rotates only the outer
        ring around an axis in the xy plane, so outer nodes move up and down on
        a sphere while the inner ring stays planar.
    outer_3d_axis_angle:
        Direction of the tilt axis in the xy plane, in radians. It is interpreted
        relative to the graph/global rotation, so the tilt direction moves with
        the graph if `global_rotation` is used.
    global_rotation:
        A rotation applied to the graph in the xy plane, in radians. This prevents
        models from relying on absolute in-plane orientation.
    class_phase_offset_fraction:
        Optional backward-compatible angular phase for label 1 as a fraction of
        one inner-ring sector. The default 0.0 means both classes use the same
        angular phase; keep this at 0.0 when the class should be determined by
        inner-to-outer distance rather than angle.
    """

    if label not in (0, 1):
        raise ValueError(f"label must be 0 or 1, got {label}.")
    if outer_radius <= inner_radius:
        raise ValueError(
            f"outer_radius must be larger than inner_radius, got {outer_radius} <= {inner_radius}."
        )

    dtype = _as_dtype(dtype)
    inner_step = 2.0 * pi / float(n_inner)
    class_phase_offset = label * class_phase_offset_fraction * inner_step
    outer_phase = outer_rotation_clockwise + class_phase_offset

    center = torch.zeros((1, 3), dtype=dtype)
    inner = ring_positions(
        n_inner,
        inner_radius,
        phase_clockwise=0.0,
        global_rotation=global_rotation,
        dtype=dtype,
    )
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
    positions = torch.cat([center, inner, outer], dim=0)
    n_nodes = int(positions.shape[0])
    closest_inner_outer_distances = closest_inner_outer_distances_from_positions(
        positions,
        n_inner=n_inner,
        n_outer=n_outer,
    )
    closest_inner_outer_summary = _distance_summary(closest_inner_outer_distances)

    species = torch.ones(n_nodes, dtype=torch.long)
    node_role = torch.tensor(
        [CENTER_ROLE] + [INNER_ROLE] * n_inner + [OUTER_ROLE] * n_outer,
        dtype=torch.long,
    )
    edge_index = ring_graph_edge_index(
        n_inner=n_inner,
        n_outer=n_outer,
        add_inner_ring_edges=add_inner_ring_edges,
        add_outer_ring_edges=add_outer_ring_edges,
        add_center_outer_edges=add_center_outer_edges,
    )

    sample_metadata: dict[str, Any] = {
        "class_name": RING_GRAPH_CLASS_NAMES[label],
        "n_inner": int(n_inner),
        "n_outer": int(n_outer),
        "inner_radius": float(inner_radius),
        "outer_radius": float(outer_radius),
        "outer_gap": float(outer_radius - inner_radius),
        "closest_inner_outer_distances": [float(x) for x in closest_inner_outer_distances.tolist()],
        "closest_inner_outer_distance_min": closest_inner_outer_summary["min"],
        "closest_inner_outer_distance_mean": closest_inner_outer_summary["mean"],
        "closest_inner_outer_distance_max": closest_inner_outer_summary["max"],
        "outer_rotation_clockwise": float(outer_rotation_clockwise),
        "outer_3d_rotation": float(outer_3d_rotation),
        "outer_3d_axis_angle": float(outer_3d_axis_angle),
        "outer_3d_rotation_deg": float(outer_3d_rotation * 180.0 / pi),
        "outer_3d_axis_deg": float(outer_3d_axis_angle * 180.0 / pi),
        "global_rotation": float(global_rotation),
        "class_phase_offset_clockwise": float(class_phase_offset),
        "outer_phase_clockwise": float(outer_phase),
        "class_phase_offset_fraction": float(class_phase_offset_fraction),
        "add_inner_ring_edges": bool(add_inner_ring_edges),
        "add_outer_ring_edges": bool(add_outer_ring_edges),
        "add_center_outer_edges": bool(add_center_outer_edges),
    }
    if metadata:
        sample_metadata.update(dict(metadata))

    return RingGraphEnvironment(
        name=name or f"ring_graph_{RING_GRAPH_CLASS_NAMES[label]}",
        label=label,
        Z=species,
        R=positions,
        edge_index=edge_index,
        node_role=node_role,
        central_atom_local_index=0,
        metadata=sample_metadata,
    )


def create_rotating_ring_dataset(
    *,
    n_graphs: int = 500,
    seed: int = 0,
    n_inner: int = 8,
    n_outer: int = 8,
    inner_radius_range: tuple[float, float] = (1.0, 1.0),
    outer_gap_range: tuple[float, float] = (1.2, 1.2),
    outer_rotation_fraction_range: tuple[float, float] = (0.0, 0.45),
    outer_3d_rotation_range: tuple[float, float] = (0.0, 0.0),
    outer_3d_axis_angle: float = 0.0,
    global_rotation_fraction_range: tuple[float, float] = (0.0, 0.0),
    distance_split_statistic: str = "mean",
    smooth_order: bool = True,
    shuffle: bool = False,
    add_inner_ring_edges: bool = False,
    add_outer_ring_edges: bool = False,
    add_center_outer_edges: bool = False,
    dtype: torch.dtype | None = None,
) -> list[RingGraphEnvironment]:
    """Generate a balanced two-class rotating-ring dataset by distance split.

    Every generated graph uses the same `inner_radius_range` and the same
    `outer_gap_range`. There are no class-specific outer gaps or radii.

    Labels are assigned after all candidate geometries are generated:
        * sort graphs by a realized closest inner-to-outer distance statistic;
        * assign the lower half to label 0 / "close";
        * assign the upper half to label 1 / "far".

    If `n_graphs` is odd, the middle graph in the sorted distance histogram is
    discarded, so the returned dataset has `2 * floor(n_graphs / 2)` graphs.

    `distance_split_statistic` can be "min", "mean", or "max" and selects which
    per-graph closest-distance summary is used for sorting. The default "mean"
    is usually the most stable way to split by the whole closest-distance
    distribution while still producing one scalar per graph.

    `outer_rotation_fraction_range` is expressed as a fraction of one inner-ring
    sector. With the default n_inner=8, one sector is 45 degrees, so the outer
    ring rotates from 0 to 20.25 degrees. The generated labels are not chosen
    directly from this angle; angle only changes the realized closest distances,
    and the sorted distance histogram determines the labels.

    `outer_3d_rotation_range` is in radians. Its default is (0, 0), so the
    dataset is 2D. Setting it to, for example, (0, pi / 3) makes the outer ring
    smoothly tilt from planar to 60 degrees while the inner ring remains in the
    xy plane.
    """

    if n_graphs <= 1:
        raise ValueError(f"n_graphs must be at least 2 for a balanced split, got {n_graphs}.")
    if n_inner <= 0 or n_outer <= 0:
        raise ValueError(f"n_inner and n_outer must be positive, got {n_inner}, {n_outer}.")
    if distance_split_statistic not in {"min", "mean", "max"}:
        raise ValueError(
            "distance_split_statistic must be one of {'min', 'mean', 'max'}, "
            f"got {distance_split_statistic!r}."
        )

    dtype = _as_dtype(dtype)
    generator = torch.Generator().manual_seed(int(seed))
    inner_step = 2.0 * pi / float(n_inner)

    def fraction_for_index(index: int, count: int) -> float:
        return 0.0 if count <= 1 else float(index) / float(count - 1)

    def lerp(value_range: tuple[float, float], t: float) -> float:
        low, high = value_range
        if high < low:
            raise ValueError(f"Expected range high >= low, got {value_range}.")
        return float(low + (high - low) * t)

    for range_name, value_range in (
        ("inner_radius_range", inner_radius_range),
        ("outer_gap_range", outer_gap_range),
        ("outer_rotation_fraction_range", outer_rotation_fraction_range),
        ("outer_3d_rotation_range", outer_3d_rotation_range),
        ("global_rotation_fraction_range", global_rotation_fraction_range),
    ):
        low, high = value_range
        if high < low:
            raise ValueError(f"Expected {range_name} high >= low, got {value_range}.")

    candidates: list[RingGraphEnvironment] = []
    for candidate_index in range(n_graphs):
        variation_t = fraction_for_index(candidate_index, n_graphs)

        if smooth_order:
            inner_radius = lerp(inner_radius_range, variation_t)
            outer_gap = lerp(outer_gap_range, variation_t)
            outer_rotation_fraction = lerp(outer_rotation_fraction_range, variation_t)
            outer_3d_rotation = lerp(outer_3d_rotation_range, variation_t)
            global_rotation_fraction = lerp(global_rotation_fraction_range, variation_t)
        else:
            inner_radius = _uniform(generator, *inner_radius_range)
            outer_gap = _uniform(generator, *outer_gap_range)
            outer_rotation_fraction = _uniform(generator, *outer_rotation_fraction_range)
            outer_3d_rotation = _uniform(generator, *outer_3d_rotation_range)
            global_rotation_fraction = _uniform(generator, *global_rotation_fraction_range)

        outer_radius = inner_radius + outer_gap
        outer_rotation_clockwise = outer_rotation_fraction * inner_step
        global_rotation = global_rotation_fraction * inner_step

        env = create_ring_graph_environment(
            label=0,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            outer_rotation_clockwise=outer_rotation_clockwise,
            outer_3d_rotation=outer_3d_rotation,
            outer_3d_axis_angle=outer_3d_axis_angle,
            global_rotation=global_rotation,
            n_inner=n_inner,
            n_outer=n_outer,
            class_phase_offset_fraction=0.0,
            name=f"rotating_ring_unlabeled_{candidate_index:04d}",
            add_inner_ring_edges=add_inner_ring_edges,
            add_outer_ring_edges=add_outer_ring_edges,
            add_center_outer_edges=add_center_outer_edges,
            dtype=dtype,
            metadata={
                "seed": int(seed),
                "candidate_index": int(candidate_index),
                "generation_index": int(candidate_index),
                "variation_t": float(variation_t),
                "smooth_order": bool(smooth_order),
                "outer_rotation_fraction": float(outer_rotation_fraction),
                "outer_3d_rotation_range": tuple(float(x) for x in outer_3d_rotation_range),
                "outer_3d_rotation_deg_range": tuple(float(x) * 180.0 / pi for x in outer_3d_rotation_range),
                "outer_3d_axis_angle": float(outer_3d_axis_angle),
                "outer_3d_axis_deg": float(outer_3d_axis_angle * 180.0 / pi),
                "global_rotation_fraction": float(global_rotation_fraction),
                "class_distance_mode": "distance_histogram_split",
                "distance_split_statistic": distance_split_statistic,
                "inner_radius_range": tuple(float(x) for x in inner_radius_range),
                "outer_gap_range": tuple(float(x) for x in outer_gap_range),
                "outer_rotation_fraction_range": tuple(float(x) for x in outer_rotation_fraction_range),
                "global_rotation_fraction_range": tuple(float(x) for x in global_rotation_fraction_range),
            },
        )
        candidates.append(env)

    statistic_key = f"closest_inner_outer_distance_{distance_split_statistic}"
    sorted_candidates = sorted(
        candidates,
        key=lambda env: (float(env.metadata[statistic_key]), int(env.metadata["candidate_index"])),
    )

    per_class_count = n_graphs // 2
    close_candidates = sorted_candidates[:per_class_count]
    far_candidates = sorted_candidates[-per_class_count:]
    kept_by_candidate_index = {int(env.metadata["candidate_index"]): env for env in close_candidates + far_candidates}
    split_threshold_low = float(close_candidates[-1].metadata[statistic_key])
    split_threshold_high = float(far_candidates[0].metadata[statistic_key])

    relabeled_by_candidate_index: dict[int, RingGraphEnvironment] = {}
    for label, class_candidates in enumerate((close_candidates, far_candidates)):
        for class_index, env in enumerate(class_candidates):
            candidate_index = int(env.metadata["candidate_index"])
            metadata = dict(env.metadata)
            metadata.update(
                {
                    "class_name": RING_GRAPH_CLASS_NAMES[label],
                    "class_index": int(class_index),
                    "balanced_class_count": int(per_class_count),
                    "requested_n_graphs": int(n_graphs),
                    "returned_n_graphs": int(2 * per_class_count),
                    "dropped_for_balance": int(n_graphs - 2 * per_class_count),
                    "distance_split_rank": int(sorted_candidates.index(env)),
                    "distance_split_threshold_low": split_threshold_low,
                    "distance_split_threshold_high": split_threshold_high,
                    "active_outer_gap_range": tuple(float(x) for x in outer_gap_range),
                }
            )
            relabeled_by_candidate_index[candidate_index] = replace(
                env,
                label=label,
                name=f"rotating_ring_{RING_GRAPH_CLASS_NAMES[label]}_{class_index:04d}",
                metadata=metadata,
            )

    environments = [relabeled_by_candidate_index[i] for i in sorted(kept_by_candidate_index)]

    if shuffle:
        order = torch.randperm(len(environments), generator=generator).tolist()
        environments = [environments[i] for i in order]

    return environments


# -----------------------------------------------------------------------------
# Batch conversion helpers, similar in spirit to the provided dataset helper file
# -----------------------------------------------------------------------------


def as_padded_ring_arrays(
    environments: list[RingGraphEnvironment],
    *,
    center_on_central_node: bool = True,
) -> dict[str, torch.Tensor]:
    """Stack ring graph environments into padded tensors.

    Returns keys compatible with the style of the provided helper file:
        Z, R, T, edge_index, edge_indices

    It also returns:
        node_role, node_mask, graph_index, node_counts

    `edge_index` is a compressed atom-indexed edge list over the real atoms in
    all systems. It does not include padding nodes. `edge_indices` is padded per
    graph for HIP-NN's predefined-neighbor input.
    """

    if not environments:
        raise ValueError("Cannot stack an empty environment list.")

    max_nodes = max(env.n_nodes for env in environments)
    n_systems = len(environments)
    dtype = environments[0].R.dtype

    species = torch.zeros((n_systems, max_nodes), dtype=torch.long)
    positions = torch.zeros((n_systems, max_nodes, 3), dtype=dtype)
    node_role = torch.full((n_systems, max_nodes), PADDING_ROLE, dtype=torch.long)
    node_mask = torch.zeros((n_systems, max_nodes), dtype=torch.get_default_dtype())
    labels = torch.empty((n_systems, 1), dtype=torch.get_default_dtype())
    node_counts = torch.empty((n_systems,), dtype=torch.long)
    max_edges = max(env.edge_index.shape[1] for env in environments)
    edge_indices = torch.full((n_systems, 2, max_edges), -1, dtype=torch.long)
    graph_index_parts: list[torch.Tensor] = []
    edge_index_parts: list[torch.Tensor] = []

    atom_offset = 0
    for sample_index, env in enumerate(environments):
        n_nodes = env.n_nodes
        pos = env.R.clone()
        if center_on_central_node:
            pos = pos - pos[env.central_atom_local_index : env.central_atom_local_index + 1]

        species[sample_index, :n_nodes] = env.Z
        positions[sample_index, :n_nodes] = pos
        node_role[sample_index, :n_nodes] = env.node_role
        node_mask[sample_index, :n_nodes] = 1.0
        labels[sample_index, 0] = float(env.label)
        node_counts[sample_index] = n_nodes
        edge_indices[sample_index, :, : env.edge_index.shape[1]] = env.edge_index

        edge_index_parts.append(env.edge_index + atom_offset)
        graph_index_parts.append(torch.full((n_nodes,), sample_index, dtype=torch.long))
        atom_offset += n_nodes

    edge_index = torch.cat(edge_index_parts, dim=1) if edge_index_parts else torch.empty((2, 0), dtype=torch.long)
    graph_index = torch.cat(graph_index_parts, dim=0) if graph_index_parts else torch.empty((0,), dtype=torch.long)

    return {
        "Z": species,
        "R": positions,
        "T": labels,
        "edge_index": edge_index,
        "edge_indices": edge_indices,
        "node_role": node_role,
        "node_mask": node_mask,
        "graph_index": graph_index,
        "node_counts": node_counts,
    }


def as_hippynn_arrays(
    environments: list[RingGraphEnvironment],
    *,
    center: bool = True,
) -> dict[str, torch.Tensor]:
    """Alias with a familiar name from the provided dataset helper."""

    return as_padded_ring_arrays(environments, center_on_central_node=center)


def as_pyg_data_list(environments: list[RingGraphEnvironment]) -> list[Any]:
    """Convert to torch_geometric.data.Data objects if PyTorch Geometric is installed."""

    try:
        from torch_geometric.data import Data
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install torch_geometric to use as_pyg_data_list().") from exc

    data_list = []
    for env in environments:
        data_list.append(
            Data(
                z=env.Z.clone(),
                pos=env.R.clone(),
                edge_index=env.edge_index.clone(),
                y=torch.tensor([env.label], dtype=torch.long),
                node_role=env.node_role.clone(),
                name=env.name,
                metadata=dict(env.metadata),
            )
        )
    return data_list


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a rotating-ring HTML viewer.")
    parser.add_argument("--n-graphs", type=int, default=500, help="Total number of graphs across both classes.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--n-inner", type=int, default=8, help="Number of inner-ring nodes.")
    parser.add_argument("--n-outer", type=int, default=8, help="Number of outer-ring nodes. Kept constant across samples.")
    parser.add_argument("--html", type=Path, default=DEFAULT_VISUALIZATION_DIR / "rotating_ring_viewer.html", help="Output HTML viewer path.")
    parser.add_argument("--viewer-max-graphs", type=int, default=500, help="Max graphs included in the HTML slider.")
    parser.add_argument("--inner-radius-min", type=float, default=1.0)
    parser.add_argument("--inner-radius-max", type=float, default=1.0)
    parser.add_argument("--outer-gap-min", type=float, default=1.2, help="Shared outer gap for both labels.")
    parser.add_argument("--outer-gap-max", type=float, default=1.2, help="Shared outer gap for both labels.")
    parser.add_argument("--distance-split-statistic", choices=("min", "mean", "max"), default="mean", help="Closest-distance summary used to split the histogram into balanced labels.")
    parser.add_argument("--outer-rotation-frac-min", type=float, default=0.0)
    parser.add_argument("--outer-rotation-frac-max", type=float, default=0.45)
    parser.add_argument(
        "--outer-3d-rotation-deg",
        type=float,
        default=None,
        help=(
            "Convenience value for the maximum out-of-plane outer-ring tilt in degrees. "
            "With the default smooth ordering, samples sweep from --outer-3d-rotation-deg-min "
            "to this value. Use 0 to keep the dataset 2D."
        ),
    )
    parser.add_argument("--outer-3d-rotation-deg-min", type=float, default=0.0)
    parser.add_argument("--outer-3d-rotation-deg-max", type=float, default=0.0)
    parser.add_argument(
        "--outer-3d-axis-deg",
        type=float,
        default=0.0,
        help="In-plane direction of the outer-ring tilt axis in degrees. 0 means the +x axis.",
    )
    parser.add_argument("--global-rotation-frac-min", type=float, default=0.0)
    parser.add_argument("--global-rotation-frac-max", type=float, default=0.0)
    parser.add_argument(
        "--random-parameters",
        action="store_true",
        help="Sample graph parameters randomly instead of along the smooth ordered path.",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle graph storage order after generation.")
    parser.add_argument("--add-inner-ring-edges", action="store_true")
    parser.add_argument("--add-outer-ring-edges", action="store_true")
    parser.add_argument("--add-center-outer-edges", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outer_3d_rotation_deg_max = (
        args.outer_3d_rotation_deg
        if args.outer_3d_rotation_deg is not None
        else args.outer_3d_rotation_deg_max
    )
    envs = create_rotating_ring_dataset(
        n_graphs=args.n_graphs,
        seed=args.seed,
        n_inner=args.n_inner,
        n_outer=args.n_outer,
        inner_radius_range=(args.inner_radius_min, args.inner_radius_max),
        outer_gap_range=(args.outer_gap_min, args.outer_gap_max),
        outer_rotation_fraction_range=(args.outer_rotation_frac_min, args.outer_rotation_frac_max),
        outer_3d_rotation_range=(
            args.outer_3d_rotation_deg_min * pi / 180.0,
            outer_3d_rotation_deg_max * pi / 180.0,
        ),
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
