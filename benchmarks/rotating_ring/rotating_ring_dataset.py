"""Rotating-ring graph dataset and Plotly HTML viewer.

This file creates a two-class synthetic graph dataset similar to the sketch in the
prompt: one central node, one inner ring, and one outer ring in 3D coordinates.
By default both rings are planar (z=0), but the outer ring can optionally be
tilted out of the xy plane while the inner ring remains planar.

Default class definitions
-------------------------
label 0, "aligned":
    the outer ring is close to the same angular spokes as the inner ring.

label 1, "interleaved":
    the outer ring has the same per-sample rotation as label 0, plus a half-spoke
    offset. The outer nodes sit between the inner spokes.

Both classes sample from the same radius ranges, the same global rotation range,
the same outer-ring rotation range, and the same optional 3D outer-ring tilt
range. The class label is therefore encoded in relative ring geometry, not
absolute orientation, 3D tilt, or node count.

The topology is undirected and, by default, contains center-to-inner edges and
inner-to-outer spoke edges. Ring-cycle edges can be enabled with flags if desired.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from html import escape
from math import pi
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


RING_GRAPH_CLASS_NAMES = ("aligned", "interleaved")
VIEWER_TEMPLATE_VERSION = "class-filtered-smooth-outer3d-camera-v4"
VIEWER_VERSION = "class-filtered-smooth-slider-outer3d-camera-v4"

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
        Integer class label. 0 = aligned, 1 = interleaved by default.
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
    """Build the default undirected topology for a ring graph.

    Node indexing convention:
        0                         center
        1 .. n_inner              inner ring
        1 + n_inner .. end        outer ring

    For n_outer != n_inner, each outer node is connected to the nearest matching
    inner index by fractional progress around the circle. The default dataset
    keeps n_outer constant and equal to n_inner, but this makes future changes
    straightforward.
    """

    if n_inner <= 0:
        raise ValueError(f"n_inner must be positive, got {n_inner}.")
    if n_outer <= 0:
        raise ValueError(f"n_outer must be positive, got {n_outer}.")

    center = 0
    inner_start = 1
    outer_start = 1 + n_inner
    edges: list[tuple[int, int]] = []

    # Spokes from center to inner ring.
    for i in range(n_inner):
        edges.append((center, inner_start + i))

    # Spokes from inner ring to outer ring.
    for j in range(n_outer):
        mapped_inner = int(round(j * n_inner / n_outer)) % n_inner
        edges.append((inner_start + mapped_inner, outer_start + j))

    if add_center_outer_edges:
        for j in range(n_outer):
            edges.append((center, outer_start + j))

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
    class_phase_offset_fraction: float = 0.5,
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
        0 for aligned, 1 for interleaved.
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
        Additional phase for label 1 as a fraction of one inner-ring sector.
        The default 0.5 places outer nodes halfway between inner spokes.
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
    inner_radius_range: tuple[float, float] = (1.0, 1.8),
    outer_gap_range: tuple[float, float] = (0.8, 1.6),
    outer_rotation_fraction_range: tuple[float, float] = (0.0, 0.45),
    outer_3d_rotation_range: tuple[float, float] = (0.0, 0.0),
    outer_3d_axis_angle: float = 0.0,
    global_rotation_fraction_range: tuple[float, float] = (0.0, 0.0),
    class_phase_offset_fraction: float = 0.5,
    smooth_order: bool = True,
    shuffle: bool = False,
    add_inner_ring_edges: bool = False,
    add_outer_ring_edges: bool = False,
    add_center_outer_edges: bool = False,
    dtype: torch.dtype | None = None,
) -> list[RingGraphEnvironment]:
    """Generate a balanced two-class rotating-ring dataset.

    `n_graphs=500` creates 250 examples of each class. If n_graphs is odd,
    label 1 receives the extra graph.

    By default, the dataset is not shuffled. It is ordered by class and then by
    a smooth variation coordinate `t` from 0 to 1. Adjacent examples in a class
    therefore differ only slightly in ring radii and outer-ring rotation, which
    makes the HTML slider useful for visual inspection. Set `smooth_order=False`
    for the older random parameter sampling, and set `shuffle=True` only if you
    explicitly want randomized storage order.

    `outer_rotation_fraction_range` is expressed as a fraction of one inner-ring
    sector. With the default n_inner=8, one sector is 45 degrees, so the outer
    ring rotates from 0 to 20.25 degrees for label 0 and from 22.5 to 42.75
    degrees for label 1 after the class offset is added.

    `outer_3d_rotation_range` is in radians. Its default is (0, 0), so the
    dataset is 2D exactly like the earlier version. Setting it to, for example,
    (0, pi / 3) makes the outer ring smoothly tilt from planar to 60 degrees
    while the inner ring remains in the xy plane.

    `outer_3d_axis_angle` is the in-plane direction of the tilt axis in radians.
    Its default is 0, which tilts around the +x axis.

    `global_rotation_fraction_range` is also expressed as a fraction of one
    inner-ring sector. Its default is (0, 0), so only the outer ring rotates
    relative to the inner ring in the generated viewer. You can set it to, for
    example, (0, 8) to make the whole graph smoothly sweep through 360 degrees.
    """

    if n_graphs <= 0:
        raise ValueError(f"n_graphs must be positive, got {n_graphs}.")
    if n_inner <= 0 or n_outer <= 0:
        raise ValueError(f"n_inner and n_outer must be positive, got {n_inner}, {n_outer}.")

    dtype = _as_dtype(dtype)
    generator = torch.Generator().manual_seed(int(seed))
    counts = [n_graphs // 2, n_graphs - (n_graphs // 2)]
    inner_step = 2.0 * pi / float(n_inner)

    def fraction_for_index(index: int, count: int) -> float:
        return 0.0 if count <= 1 else float(index) / float(count - 1)

    def lerp(value_range: tuple[float, float], t: float) -> float:
        low, high = value_range
        if high < low:
            raise ValueError(f"Expected range high >= low, got {value_range}.")
        return float(low + (high - low) * t)

    environments: list[RingGraphEnvironment] = []
    running_index = 0
    for label, count in enumerate(counts):
        for class_index in range(count):
            variation_t = fraction_for_index(class_index, count)

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
                label=label,
                inner_radius=inner_radius,
                outer_radius=outer_radius,
                outer_rotation_clockwise=outer_rotation_clockwise,
                outer_3d_rotation=outer_3d_rotation,
                outer_3d_axis_angle=outer_3d_axis_angle,
                global_rotation=global_rotation,
                n_inner=n_inner,
                n_outer=n_outer,
                class_phase_offset_fraction=class_phase_offset_fraction,
                name=f"rotating_ring_{RING_GRAPH_CLASS_NAMES[label]}_{class_index:04d}",
                add_inner_ring_edges=add_inner_ring_edges,
                add_outer_ring_edges=add_outer_ring_edges,
                add_center_outer_edges=add_center_outer_edges,
                dtype=dtype,
                metadata={
                    "seed": int(seed),
                    "class_index": int(class_index),
                    "generation_index": int(running_index),
                    "variation_t": float(variation_t),
                    "smooth_order": bool(smooth_order),
                    "outer_rotation_fraction": float(outer_rotation_fraction),
                    "outer_3d_rotation_range": tuple(float(x) for x in outer_3d_rotation_range),
                    "outer_3d_rotation_deg_range": tuple(float(x) * 180.0 / pi for x in outer_3d_rotation_range),
                    "outer_3d_axis_angle": float(outer_3d_axis_angle),
                    "outer_3d_axis_deg": float(outer_3d_axis_angle * 180.0 / pi),
                    "global_rotation_fraction": float(global_rotation_fraction),
                    "inner_radius_range": tuple(float(x) for x in inner_radius_range),
                    "outer_gap_range": tuple(float(x) for x in outer_gap_range),
                    "outer_rotation_fraction_range": tuple(float(x) for x in outer_rotation_fraction_range),
                    "global_rotation_fraction_range": tuple(float(x) for x in global_rotation_fraction_range),
                },
            )
            environments.append(env)
            running_index += 1

    if shuffle:
        order = torch.randperm(len(environments), generator=generator).tolist()
        environments = [environments[i] for i in order]

    return environments


# -----------------------------------------------------------------------------
# Batch conversion helpers, similar in spirit to the provided dataset helper file
# -----------------------------------------------------------------------------


def atom_mask_from_local(species: torch.Tensor, local_indices: torch.Tensor | int) -> torch.Tensor:
    """Build a padded atom mask with one selected local atom per system."""

    if species.ndim != 2:
        raise ValueError(f"Expected species to have shape [n_systems, n_atoms_max], got {tuple(species.shape)}.")

    n_systems, n_atoms_max = species.shape
    local_indices = torch.as_tensor(local_indices, dtype=torch.long, device=species.device)
    if local_indices.ndim == 0:
        local_indices = local_indices.expand(n_systems)
    if local_indices.shape[0] != n_systems:
        raise ValueError(
            f"Expected central atom local indices for {n_systems} systems, got shape {tuple(local_indices.shape)}."
        )
    if (local_indices < 0).any() or (local_indices >= n_atoms_max).any():
        raise ValueError("Central atom local indices are out of bounds for the padded atom dimension.")

    system_indices = torch.arange(n_systems, dtype=torch.long, device=species.device)
    if (species[system_indices, local_indices] == 0).any():
        raise ValueError("Central atom local indices must point to real, non-padding atoms.")

    atom_mask = torch.zeros_like(species, dtype=torch.get_default_dtype())
    atom_mask[system_indices, local_indices] = 1.0
    return atom_mask


def as_padded_ring_arrays(
    environments: list[RingGraphEnvironment],
    *,
    center_on_central_node: bool = True,
) -> dict[str, torch.Tensor]:
    """Stack ring graph environments into padded tensors.

    Returns keys compatible with the style of the provided helper file:
        Z, R, T, central_atom_mask, edge_index

    It also returns:
        node_role, node_mask, graph_index, node_counts

    `edge_index` is a compressed atom-indexed edge list over the real atoms in
    all systems. It does not include padding nodes.
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
    central_indices = torch.empty((n_systems,), dtype=torch.long)
    node_counts = torch.empty((n_systems,), dtype=torch.long)
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
        central_indices[sample_index] = int(env.central_atom_local_index)
        node_counts[sample_index] = n_nodes

        edge_index_parts.append(env.edge_index + atom_offset)
        graph_index_parts.append(torch.full((n_nodes,), sample_index, dtype=torch.long))
        atom_offset += n_nodes

    edge_index = torch.cat(edge_index_parts, dim=1) if edge_index_parts else torch.empty((2, 0), dtype=torch.long)
    graph_index = torch.cat(graph_index_parts, dim=0) if graph_index_parts else torch.empty((0,), dtype=torch.long)

    return {
        "Z": species,
        "R": positions,
        "T": labels,
        "central_atom_mask": atom_mask_from_local(species, central_indices),
        "edge_index": edge_index,
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


def save_ring_dataset(
    environments: list[RingGraphEnvironment],
    path: str | Path,
    *,
    center_on_central_node: bool = True,
) -> None:
    """Save padded tensors plus metadata to a torch .pt file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "rotating_ring_graph_dataset_v3_outer_3d",
        "class_names": RING_GRAPH_CLASS_NAMES,
        "arrays": as_padded_ring_arrays(environments, center_on_central_node=center_on_central_node),
        "metadata": [
            {
                "name": env.name,
                "label": int(env.label),
                **dict(env.metadata),
            }
            for env in environments
        ],
    }
    torch.save(payload, path)


def load_ring_dataset(path: str | Path) -> dict[str, Any]:
    """Load a dataset saved by save_ring_dataset()."""

    return torch.load(Path(path), map_location="cpu", weights_only=False)


# -----------------------------------------------------------------------------
# Plotly HTML viewer
# -----------------------------------------------------------------------------


def _edge_trace_for_env(env: RingGraphEnvironment) -> Any:
    import plotly.graph_objects as go

    x: list[float | None] = []
    y: list[float | None] = []
    z: list[float | None] = []
    for src, dst in undirected_edge_pairs(env.edge_index):
        x.extend([float(env.R[src, 0]), float(env.R[dst, 0]), None])
        y.extend([float(env.R[src, 1]), float(env.R[dst, 1]), None])
        z.extend([float(env.R[src, 2]), float(env.R[dst, 2]), None])

    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line={"width": 5, "color": "rgba(120,120,120,0.75)"},
        hoverinfo="skip",
        showlegend=False,
        name="edges",
    )


def _node_trace_for_env(env: RingGraphEnvironment) -> Any:
    import plotly.graph_objects as go

    role_to_color = {
        CENTER_ROLE: "#f47c20",
        INNER_ROLE: "#145f7a",
        OUTER_ROLE: "#44a9cc",
    }
    role_to_name = {
        CENTER_ROLE: "center",
        INNER_ROLE: "inner",
        OUTER_ROLE: "outer",
    }
    colors = [role_to_color[int(role)] for role in env.node_role.tolist()]
    hover = [
        f"node {i}<br>role={role_to_name[int(role)]}<br>x={env.R[i,0]:.3f}<br>y={env.R[i,1]:.3f}<br>z={env.R[i,2]:.3f}"
        for i, role in enumerate(env.node_role.tolist())
    ]

    return go.Scatter3d(
        x=env.R[:, 0].tolist(),
        y=env.R[:, 1].tolist(),
        z=env.R[:, 2].tolist(),
        mode="markers",
        marker={
            "size": 9,
            "color": colors,
            "line": {"width": 1.5, "color": "#202020"},
        },
        text=[str(i) for i in range(env.n_nodes)],
        hovertext=hover,
        hoverinfo="text",
        showlegend=False,
        name="nodes",
    )


def _viewer_title(env: RingGraphEnvironment, index: int, total: int) -> str:
    meta = dict(env.metadata)
    deg = 180.0 / pi
    return (
        f"Graph {index + 1}/{total}: {env.name} | "
        f"label={env.label} ({meta.get('class_name', 'unknown')}) | "
        f"r_inner={meta.get('inner_radius', float('nan')):.3f}, "
        f"r_outer={meta.get('outer_radius', float('nan')):.3f}, "
        f"outer phase={meta.get('outer_phase_clockwise', float('nan')) * deg:.1f} deg, "
        f"outer 3D tilt={meta.get('outer_3d_rotation', 0.0) * deg:.1f} deg"
    )


def _validate_generated_viewer_html(html: str) -> None:
    """Fail fast if this file ever emits the older single-slider viewer."""

    required_markers = [
        VIEWER_VERSION,
        'id="class-buttons"',
        'graphsByClass',
        'setActiveClass',
        'The slider is ordered by the smooth variation coordinate inside the selected class',
        'outer3DRotationDeg',
        'getCurrentCamera',
        'rememberCurrentCamera',
        'scene.camera',
        'uirevision',
    ]
    missing = [marker for marker in required_markers if marker not in html]
    if missing:
        raise RuntimeError(
            "Generated HTML is missing the updated class-filtered viewer markers: "
            + ", ".join(missing)
        )


def write_ring_graph_viewer(
    environments: list[RingGraphEnvironment],
    html_path: str | Path,
    *,
    max_graphs: int | None = 500,
    include_plotlyjs: bool | str = True,
) -> None:
    """Write an interactive 3D Plotly viewer with class filtering.

    The viewer keeps the classes separate. The class buttons switch between
    Class 1 (label 0) and Class 2 (label 1), and the slider then walks only
    through the selected class. Within each class, graphs are sorted by
    `metadata["variation_t"]` when available, so neighboring slider positions
    correspond to neighboring points along the same smooth parameter path.
    """

    if not environments:
        raise ValueError("Cannot write a viewer for an empty environment list.")

    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)

    def sort_value(item: tuple[int, RingGraphEnvironment]) -> tuple[float, int]:
        original_index, env = item
        meta = dict(env.metadata)
        for key in ("variation_t", "class_index", "generation_index"):
            if key in meta:
                try:
                    return float(meta[key]), int(original_index)
                except (TypeError, ValueError):
                    pass
        return float(original_index), int(original_index)

    items_by_label: dict[int, list[tuple[int, RingGraphEnvironment]]] = {}
    for original_index, env in enumerate(environments):
        items_by_label.setdefault(int(env.label), []).append((original_index, env))

    labels = sorted(items_by_label)
    for label in labels:
        items_by_label[label].sort(key=sort_value)

    # Keep the viewer balanced across classes when max_graphs is smaller than
    # the full dataset. This preserves the class switch even for small previews.
    selected_by_label: dict[int, list[tuple[int, RingGraphEnvironment]]] = {}
    if max_graphs is None:
        selected_by_label = {label: list(items_by_label[label]) for label in labels}
    else:
        requested = max(1, int(max_graphs))
        per_label = max(1, requested // max(1, len(labels)))
        remainder = max(0, requested - per_label * len(labels))
        for label_i, label in enumerate(labels):
            limit = per_label + (1 if label_i < remainder else 0)
            selected_by_label[label] = list(items_by_label[label][:limit])

    selected_items = [item for label in labels for item in selected_by_label[label]]
    if not selected_items:
        raise ValueError("No graphs selected for the viewer.")

    max_radius = max(
        float(torch.linalg.vector_norm(env.R, dim=1).max().item())
        for _, env in selected_items
    )
    axis_range = [-1.15 * max_radius, 1.15 * max_radius]
    # Use the same range for z so tilted outer-ring nodes are always visible
    # and the sphere interpretation is not visually compressed.
    z_range = list(axis_range)

    role_to_color = {
        CENTER_ROLE: "#f47c20",
        INNER_ROLE: "#145f7a",
        OUTER_ROLE: "#44a9cc",
    }
    role_to_name = {
        CENTER_ROLE: "center",
        INNER_ROLE: "inner",
        OUTER_ROLE: "outer",
    }

    records: list[dict[str, Any]] = []
    deg = 180.0 / pi
    for label in labels:
        group = selected_by_label[label]
        class_total = len(group)
        if class_total == 0:
            continue

        for class_position, (original_index, env) in enumerate(group):
            meta = dict(env.metadata)
            fallback_t = 0.0 if class_total <= 1 else class_position / float(class_total - 1)
            try:
                variation_t = float(meta.get("variation_t", fallback_t))
            except (TypeError, ValueError):
                variation_t = fallback_t

            if 0 <= label < len(RING_GRAPH_CLASS_NAMES):
                class_name = str(meta.get("class_name", RING_GRAPH_CLASS_NAMES[label]))
            else:
                class_name = str(meta.get("class_name", f"label {label}"))
            class_display = f"Class {label + 1}"

            edge_x: list[float | None] = []
            edge_y: list[float | None] = []
            edge_z: list[float | None] = []
            for src, dst in undirected_edge_pairs(env.edge_index):
                edge_x.extend([float(env.R[src, 0]), float(env.R[dst, 0]), None])
                edge_y.extend([float(env.R[src, 1]), float(env.R[dst, 1]), None])
                edge_z.extend([float(env.R[src, 2]), float(env.R[dst, 2]), None])

            node_x = [float(x) for x in env.R[:, 0].tolist()]
            node_y = [float(y) for y in env.R[:, 1].tolist()]
            node_z = [float(z) for z in env.R[:, 2].tolist()]
            node_colors = [role_to_color.get(int(role), "#999999") for role in env.node_role.tolist()]
            node_hover = []
            for node_i, role in enumerate(env.node_role.tolist()):
                role_name = role_to_name.get(int(role), f"role {int(role)}")
                node_hover.append(
                    "<br>".join(
                        [
                            f"node {node_i}",
                            f"role={role_name}",
                            f"x={float(env.R[node_i, 0]):.3f}",
                            f"y={float(env.R[node_i, 1]):.3f}",
                            f"z={float(env.R[node_i, 2]):.3f}",
                        ]
                    )
                )

            inner_radius = float(meta.get("inner_radius", float("nan")))
            outer_radius = float(meta.get("outer_radius", float("nan")))
            outer_phase = float(meta.get("outer_phase_clockwise", float("nan")))
            outer_rotation = float(meta.get("outer_rotation_clockwise", float("nan")))
            outer_3d_rotation = float(meta.get("outer_3d_rotation", 0.0))
            outer_3d_axis_angle = float(meta.get("outer_3d_axis_angle", 0.0))
            global_rotation = float(meta.get("global_rotation", float("nan")))
            title = (
                f"{class_display} ({class_name}) | variation {100.0 * variation_t:.1f}% | "
                f"graph {class_position + 1}/{class_total}<br>"
                f"r_inner={inner_radius:.3f}, r_outer={outer_radius:.3f}, "
                f"outer rotation={outer_rotation * deg:.1f} deg, "
                f"outer phase={outer_phase * deg:.1f} deg, "
                f"outer 3D tilt={outer_3d_rotation * deg:.1f} deg, "
                f"global rotation={global_rotation * deg:.1f} deg"
            )

            records.append(
                {
                    "label": int(label),
                    "classDisplay": class_display,
                    "className": class_name,
                    "classGraphIndex": int(class_position),
                    "classGraphCount": int(class_total),
                    "originalDatasetIndex": int(original_index),
                    "name": env.name,
                    "variationT": float(variation_t),
                    "title": title,
                    "innerRadius": inner_radius,
                    "outerRadius": outer_radius,
                    "outerRotationDeg": outer_rotation * deg,
                    "outerPhaseDeg": outer_phase * deg,
                    "outer3DRotationDeg": outer_3d_rotation * deg,
                    "outer3DAxisDeg": outer_3d_axis_angle * deg,
                    "globalRotationDeg": global_rotation * deg,
                    "nodeX": node_x,
                    "nodeY": node_y,
                    "nodeZ": node_z,
                    "nodeColor": node_colors,
                    "nodeHover": node_hover,
                    "edgeX": edge_x,
                    "edgeY": edge_y,
                    "edgeZ": edge_z,
                }
            )

    payload = {
        "viewerVersion": VIEWER_VERSION,
        "graphs": records,
        "axisRange": axis_range,
        "zRange": z_range,
        "viewerTemplateVersion": VIEWER_TEMPLATE_VERSION,
    }
    payload_json = json.dumps(payload, allow_nan=True)

    if include_plotlyjs is True:
        from plotly.offline import get_plotlyjs

        plotly_loader = f'<script type="text/javascript">{get_plotlyjs()}</script>'
    elif include_plotlyjs == "cdn":
        plotly_loader = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
    elif include_plotlyjs:
        plotly_loader = f'<script src="{escape(str(include_plotlyjs))}"></script>'
    else:
        plotly_loader = ""

    html = f"""<!doctype html>
<!-- rotating-ring-viewer-template: {VIEWER_VERSION} -->
<html lang="en">
<!-- viewer-version: {VIEWER_VERSION} -->
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rotating Ring Graph Viewer</title>
  <meta name="ring-viewer-version" content="{VIEWER_VERSION}">
  {plotly_loader}
  <style>
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #fafafa;
      color: #202020;
    }}
    #controls {{
      padding: 14px 18px 8px 18px;
      border-bottom: 1px solid #d6d6d6;
      background: white;
      position: sticky;
      top: 0;
      z-index: 10;
      box-shadow: 0 1px 6px rgba(0, 0, 0, 0.06);
    }}
    .control-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-bottom: 8px;
    }}
    button {{
      border: 1px solid #bbbbbb;
      background: #f5f5f5;
      border-radius: 8px;
      padding: 7px 11px;
      cursor: pointer;
      font-size: 14px;
    }}
    button.active {{
      background: #202020;
      color: white;
      border-color: #202020;
    }}
    button:hover {{
      filter: brightness(0.96);
    }}
    #graph-slider {{
      flex: 1 1 420px;
      min-width: 240px;
    }}
    #status {{
      font-size: 14px;
      font-weight: 600;
    }}
    #details {{
      font-size: 13px;
      color: #555555;
    }}
    #plot {{
      width: 100vw;
      height: calc(100vh - 118px);
      min-height: 560px;
    }}
    .hint {{
      font-size: 12px;
      color: #666666;
    }}
  </style>
</head>
<body>
  <div id="controls">
    <div class="control-row">
      <span id="class-buttons"></span>
      <button id="prev-button" type="button">Previous</button>
      <button id="next-button" type="button">Next</button>
      <button id="play-button" type="button">Play</button>
    </div>
    <div class="control-row">
      <input id="graph-slider" type="range" min="0" max="0" step="1" value="0">
      <span id="status"></span>
    </div>
    <div id="details"></div>
    <div class="hint">The slider is ordered by the smooth variation coordinate inside the selected class, so adjacent positions should show small geometry changes. Rotating or zooming the 3D view is preserved while the slider/class changes. Set outer 3D rotation to 0 for the original planar dataset, or a positive value to tilt only the outer ring. Viewer template: {VIEWER_VERSION}.</div>
  </div>
  <div id="plot"></div>

  <script type="text/javascript">
    const payload = {payload_json};
    window.RING_VIEWER_VERSION = payload.viewerVersion;
    const graphsByClass = {{}};
    for (const graph of payload.graphs) {{
      const key = String(graph.label);
      if (!graphsByClass[key]) graphsByClass[key] = [];
      graphsByClass[key].push(graph);
    }}
    for (const key of Object.keys(graphsByClass)) {{
      graphsByClass[key].sort((a, b) => a.variationT - b.variationT);
    }}

    const classLabels = Object.keys(graphsByClass).sort((a, b) => Number(a) - Number(b));
    let activeClass = classLabels[0];
    let playTimer = null;
    let storedCamera = null;

    const plotDiv = document.getElementById("plot");
    const slider = document.getElementById("graph-slider");
    const status = document.getElementById("status");
    const details = document.getElementById("details");
    const classButtons = document.getElementById("class-buttons");
    const playButton = document.getElementById("play-button");

    const config = {{responsive: true, displaylogo: false}};

    function cloneObject(value) {{
      if (value === null || value === undefined) return null;
      return JSON.parse(JSON.stringify(value));
    }}

    function getCurrentCamera() {{
      try {{
        if (plotDiv && plotDiv._fullLayout && plotDiv._fullLayout.scene && plotDiv._fullLayout.scene.camera) {{
          return cloneObject(plotDiv._fullLayout.scene.camera);
        }}
      }} catch (error) {{
        // Ignore camera reads before Plotly has finished initializing.
      }}
      return storedCamera ? cloneObject(storedCamera) : null;
    }}

    function rememberCurrentCamera() {{
      const camera = getCurrentCamera();
      if (camera) storedCamera = camera;
      return storedCamera ? cloneObject(storedCamera) : null;
    }}

    function currentClassGraphs() {{
      return graphsByClass[activeClass] || [];
    }}

    function currentSliderFraction() {{
      const graphs = currentClassGraphs();
      const maxIndex = Math.max(graphs.length - 1, 1);
      return Number(slider.value) / maxIndex;
    }}

    function makeTraces(graph) {{
      return [
        {{
          type: "scatter3d",
          mode: "lines",
          x: graph.edgeX,
          y: graph.edgeY,
          z: graph.edgeZ,
          line: {{width: 5, color: "rgba(120,120,120,0.75)"}},
          hoverinfo: "skip",
          showlegend: false,
          name: "edges"
        }},
        {{
          type: "scatter3d",
          mode: "markers",
          x: graph.nodeX,
          y: graph.nodeY,
          z: graph.nodeZ,
          marker: {{
            size: 9,
            color: graph.nodeColor,
            line: {{width: 1.5, color: "#202020"}}
          }},
          hovertext: graph.nodeHover,
          hoverinfo: "text",
          showlegend: false,
          name: "nodes"
        }}
      ];
    }}

    function makeLayout(graph, camera) {{
      const layout = {{
        title: {{text: graph.title, x: 0.02, xanchor: "left"}},
        scene: {{
          xaxis: {{title: "x", range: payload.axisRange}},
          yaxis: {{title: "y", range: payload.axisRange}},
          zaxis: {{title: "z", range: payload.zRange}},
          aspectmode: "cube",
          uirevision: "keep-camera"
        }},
        // Keep user-driven scene state, especially the 3D camera, across Plotly.react calls.
        uirevision: "keep-camera",
        margin: {{l: 0, r: 0, t: 78, b: 0}},
        showlegend: false
      }};
      if (camera) {{
        layout.scene.camera = cloneObject(camera);
      }}
      return layout;
    }}

    function refreshClassButtons() {{
      classButtons.innerHTML = "";
      for (const label of classLabels) {{
        const graphs = graphsByClass[label];
        if (!graphs || graphs.length === 0) continue;
        const first = graphs[0];
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = `${{first.classDisplay}}: ${{first.className}} (${{graphs.length}})`;
        button.className = label === activeClass ? "active" : "";
        button.addEventListener("click", () => setActiveClass(label));
        classButtons.appendChild(button);
      }}
    }}

    function setActiveClass(label) {{
      const fraction = currentSliderFraction();
      activeClass = String(label);
      const graphs = currentClassGraphs();
      const maxIndex = Math.max(graphs.length - 1, 0);
      slider.max = String(maxIndex);
      slider.value = String(Math.round(fraction * maxIndex));
      updatePlot();
    }}

    function updatePlot() {{
      const graphs = currentClassGraphs();
      if (graphs.length === 0) return;
      const maxIndex = Math.max(graphs.length - 1, 0);
      slider.max = String(maxIndex);
      let index = Number(slider.value);
      if (!Number.isFinite(index)) index = 0;
      index = Math.max(0, Math.min(maxIndex, Math.round(index)));
      slider.value = String(index);

      const graph = graphs[index];
      const camera = rememberCurrentCamera();
      Plotly.react(plotDiv, makeTraces(graph), makeLayout(graph, camera), config).then(() => {{
        if (storedCamera) {{
          Plotly.relayout(plotDiv, {{"scene.camera": cloneObject(storedCamera)}});
        }}
      }});
      status.textContent = `${{graph.classDisplay}} graph ${{index + 1}}/${{graphs.length}} | variation ${{(100 * graph.variationT).toFixed(1)}}%`;
      details.textContent = `dataset index ${{graph.originalDatasetIndex}} | ${{graph.name}} | inner radius ${{graph.innerRadius.toFixed(3)}} | outer radius ${{graph.outerRadius.toFixed(3)}} | outer rotation ${{graph.outerRotationDeg.toFixed(1)}} deg | outer phase ${{graph.outerPhaseDeg.toFixed(1)}} deg | outer 3D tilt ${{graph.outer3DRotationDeg.toFixed(1)}} deg | tilt axis ${{graph.outer3DAxisDeg.toFixed(1)}} deg`;
      refreshClassButtons();
    }}

    function stopPlay() {{
      if (playTimer !== null) {{
        window.clearInterval(playTimer);
        playTimer = null;
      }}
      playButton.textContent = "Play";
    }}

    function togglePlay() {{
      if (playTimer !== null) {{
        stopPlay();
        return;
      }}
      playButton.textContent = "Pause";
      playTimer = window.setInterval(() => {{
        const graphs = currentClassGraphs();
        const maxIndex = Math.max(graphs.length - 1, 0);
        const next = Number(slider.value) + 1;
        if (next > maxIndex) {{
          stopPlay();
          return;
        }}
        slider.value = String(next);
        updatePlot();
      }}, 130);
    }}

    slider.addEventListener("input", () => {{ stopPlay(); updatePlot(); }});
    document.getElementById("prev-button").addEventListener("click", () => {{
      stopPlay();
      slider.value = String(Math.max(0, Number(slider.value) - 1));
      updatePlot();
    }});
    document.getElementById("next-button").addEventListener("click", () => {{
      stopPlay();
      const maxIndex = Number(slider.max);
      slider.value = String(Math.min(maxIndex, Number(slider.value) + 1));
      updatePlot();
    }});
    playButton.addEventListener("click", togglePlay);

    window.addEventListener("keydown", (event) => {{
      if (event.key === "ArrowLeft") {{
        stopPlay();
        slider.value = String(Math.max(0, Number(slider.value) - 1));
        updatePlot();
      }} else if (event.key === "ArrowRight") {{
        stopPlay();
        const maxIndex = Number(slider.max);
        slider.value = String(Math.min(maxIndex, Number(slider.value) + 1));
        updatePlot();
      }}
    }});

    refreshClassButtons();
    updatePlot();
  </script>
</body>
</html>
"""

    _validate_generated_viewer_html(html)
    html_path.write_text(html, encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a rotating-ring graph dataset and optional 3D HTML viewer.")
    parser.add_argument("--n-graphs", type=int, default=500, help="Total number of graphs across both classes.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--n-inner", type=int, default=8, help="Number of inner-ring nodes.")
    parser.add_argument("--n-outer", type=int, default=8, help="Number of outer-ring nodes. Kept constant across samples.")
    parser.add_argument("--out", type=Path, default=Path("rotating_ring_dataset.pt"), help="Output torch .pt path.")
    parser.add_argument("--html", type=Path, default=Path("rotating_ring_viewer.html"), help="Output HTML viewer path.")
    parser.add_argument("--no-html", action="store_true", help="Only save the .pt dataset; do not write a viewer.")
    parser.add_argument("--viewer-max-graphs", type=int, default=500, help="Max graphs included in the HTML slider.")
    parser.add_argument("--inner-radius-min", type=float, default=1.0)
    parser.add_argument("--inner-radius-max", type=float, default=1.8)
    parser.add_argument("--outer-gap-min", type=float, default=0.8)
    parser.add_argument("--outer-gap-max", type=float, default=1.6)
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
    parser.add_argument("--class-phase-offset-frac", type=float, default=0.5)
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
        class_phase_offset_fraction=args.class_phase_offset_frac,
        smooth_order=not args.random_parameters,
        shuffle=args.shuffle,
        add_inner_ring_edges=args.add_inner_ring_edges,
        add_outer_ring_edges=args.add_outer_ring_edges,
        add_center_outer_edges=args.add_center_outer_edges,
    )
    save_ring_dataset(envs, args.out)
    print(f"saved {len(envs)} graphs to {args.out}")

    if not args.no_html:
        write_ring_graph_viewer(envs, args.html, max_graphs=args.viewer_max_graphs)
        print(f"saved viewer to {args.html} using viewer version {VIEWER_VERSION}")


if __name__ == "__main__":
    main()
