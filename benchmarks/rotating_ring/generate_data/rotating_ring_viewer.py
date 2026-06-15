"""Plotly HTML viewer for rotating-ring graph datasets."""

from __future__ import annotations

import json
from html import escape
from math import pi
from pathlib import Path
from typing import Any

import torch

try:
    from .rotating_ring_dataset import (
        CENTER_ROLE,
        INNER_ROLE,
        OUTER_ROLE,
        RING_GRAPH_CLASS_NAMES,
        RingGraphEnvironment,
        undirected_edge_pairs,
    )
except ImportError:
    from rotating_ring_dataset import (
        CENTER_ROLE,
        INNER_ROLE,
        OUTER_ROLE,
        RING_GRAPH_CLASS_NAMES,
        RingGraphEnvironment,
        undirected_edge_pairs,
    )


VIEWER_TEMPLATE_VERSION = "single-slider-smooth-outer3d-camera-distance-v11"
VIEWER_VERSION = "single-slider-smooth-outer3d-camera-distance-v11"


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


def _wrapped_angle_difference_degrees(angle_a: torch.Tensor, angle_b: torch.Tensor) -> torch.Tensor:
    """Return the absolute wrapped angular difference in degrees.

    The result is in [0, 180]. This is used as a viewer diagnostic: it reports
    the xy-plane angular gap between each inner node and its nearest outer node.
    """

    diff = torch.atan2(torch.sin(angle_a - angle_b), torch.cos(angle_a - angle_b)).abs()
    return diff * (180.0 / pi)


def _closest_inner_outer_distance_angle_pairs(env: RingGraphEnvironment) -> tuple[list[float], list[float]]:
    """Return nearest outer distance and xy-angle gap for each inner node.

    For every inner-ring node, this computes the nearest outer-ring node by full
    3D Euclidean distance. It then reports both the closest distance and the
    absolute wrapped xy-plane angular gap to that same nearest outer node.
    """

    inner_mask = env.node_role == INNER_ROLE
    outer_mask = env.node_role == OUTER_ROLE
    inner_points = env.R[inner_mask]
    outer_points = env.R[outer_mask]
    if inner_points.numel() == 0 or outer_points.numel() == 0:
        return [], []

    distances = torch.cdist(inner_points, outer_points, p=2.0)
    closest_values, closest_outer_indices = distances.min(dim=1)

    inner_angles = torch.atan2(inner_points[:, 1], inner_points[:, 0])
    nearest_outer_points = outer_points[closest_outer_indices]
    nearest_outer_angles = torch.atan2(nearest_outer_points[:, 1], nearest_outer_points[:, 0])
    angle_gaps_deg = _wrapped_angle_difference_degrees(inner_angles, nearest_outer_angles)

    return (
        [float(value) for value in closest_values.tolist()],
        [float(value) for value in angle_gaps_deg.tolist()],
    )


def _closest_inner_outer_distances(env: RingGraphEnvironment) -> list[float]:
    """Return the nearest outer-ring distance for each inner-ring node."""

    distances, _ = _closest_inner_outer_distance_angle_pairs(env)
    return distances


def _summary_from_values(values: list[float]) -> dict[str, float]:
    """Return min/mean/max summary values for a list of distances."""

    clean = [float(value) for value in values if value == value]
    if not clean:
        nan = float("nan")
        return {"min": nan, "mean": nan, "max": nan}
    return {
        "min": min(clean),
        "mean": sum(clean) / float(len(clean)),
        "max": max(clean),
    }


def _finite_float_or_nan(value: Any) -> float:
    """Return value as float, or NaN when it cannot be converted."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _validate_generated_viewer_html(html: str) -> None:
    """Fail fast if this file ever emits the older class-button viewer."""

    required_markers = [
        VIEWER_VERSION,
        'id="graph-slider"',
        'allGraphs',
        'class-boundary',
        'The single slider walks through all selected graphs',
        'outer3DRotationDeg',
        'closestInnerOuterDistances',
        'id="distance-histogram"',
        'updateDistanceHistogram',
        'distanceRange',
        'meanDistanceRange',
        'closestInnerOuterAngleDeg',
        'id="distance-angle-scatter"',
        'updateDistanceAngleScatter',
        'getCurrentCamera',
        'rememberCurrentCamera',
        'scene.camera',
        'uirevision',
        'distanceSplitCutoff',
        'selected graph average',
        'class cutoff',
    ]
    missing = [marker for marker in required_markers if marker not in html]
    if missing:
        raise RuntimeError(
            "Generated HTML is missing the updated single-slider viewer markers: "
            + ", ".join(missing)
        )


def write_ring_graph_viewer(
    environments: list[RingGraphEnvironment],
    html_path: str | Path,
    *,
    max_graphs: int | None = 500,
    include_plotlyjs: bool | str = True,
) -> None:
    """Write an interactive 3D Plotly viewer with one continuous slider.

    Graphs are still selected evenly from each class and sorted by class, then
    by the distance split order when available. The HTML viewer exposes one
    slider over that combined order; when the slider crosses from one label to
    the next, the status/header text changes to show the active class.

    The top histogram shows one value per graph: the average nearest
    inner-to-outer distance. The dotted vertical line is the class cutoff from
    the split thresholds. The dashed vertical line is the currently selected
    graph average. The bottom angle-vs-distance scatter keeps the raw per-inner
    nearest-node diagnostics.
    """

    if not environments:
        raise ValueError("Cannot write a viewer for an empty environment list.")

    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)

    def sort_value(item: tuple[int, RingGraphEnvironment]) -> tuple[float, int]:
        original_index, env = item
        meta = dict(env.metadata)

        # Prefer the split order/class order because labels are assigned from the
        # measured closest-distance statistic. This matters for mixed-count rings,
        # where the rotation parameter is not always monotonic in distance.
        for key in ("class_index", "distance_split_rank", "variation_t", "generation_index"):
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

            (
                closest_inner_outer_distances,
                closest_inner_outer_angle_deg,
            ) = _closest_inner_outer_distance_angle_pairs(env)

            closest_inner_outer_summary = _summary_from_values(closest_inner_outer_distances)
            closest_inner_outer_angle_summary = _summary_from_values(closest_inner_outer_angle_deg)

            distance_split_threshold_low = _finite_float_or_nan(
                meta.get("distance_split_threshold_low", float("nan"))
            )
            distance_split_threshold_high = _finite_float_or_nan(
                meta.get("distance_split_threshold_high", float("nan"))
            )
            if (
                distance_split_threshold_low == distance_split_threshold_low
                and distance_split_threshold_high == distance_split_threshold_high
            ):
                distance_split_cutoff = 0.5 * (
                    distance_split_threshold_low + distance_split_threshold_high
                )
            else:
                distance_split_cutoff = float("nan")

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
                    "closestInnerOuterDistances": closest_inner_outer_distances,
                    "closestInnerOuterAngleDeg": closest_inner_outer_angle_deg,
                    "closestInnerOuterDistanceMin": closest_inner_outer_summary["min"],
                    "closestInnerOuterDistanceMean": closest_inner_outer_summary["mean"],
                    "closestInnerOuterDistanceMax": closest_inner_outer_summary["max"],
                    "distanceSplitThresholdLow": distance_split_threshold_low,
                    "distanceSplitThresholdHigh": distance_split_threshold_high,
                    "distanceSplitCutoff": distance_split_cutoff,
                    "closestInnerOuterAngleMinDeg": closest_inner_outer_angle_summary["min"],
                    "closestInnerOuterAngleMeanDeg": closest_inner_outer_angle_summary["mean"],
                    "closestInnerOuterAngleMaxDeg": closest_inner_outer_angle_summary["max"],
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

    # This range is for the bottom angle-vs-distance scatter, which still shows
    # raw nearest distances for each inner node.
    all_closest_distances = [
        distance
        for record in records
        for distance in record["closestInnerOuterDistances"]
        if distance == distance
    ]
    if all_closest_distances:
        min_distance = min(all_closest_distances)
        max_distance = max(all_closest_distances)
        if max_distance > min_distance:
            distance_padding = 0.05 * (max_distance - min_distance)
        else:
            distance_padding = max(1.0e-6, 0.05 * abs(max_distance), 0.05)
        distance_range = [
            min_distance - distance_padding,
            max_distance + distance_padding,
        ]
    else:
        distance_range = [0.0, 1.0]

    # This range is for the top histogram. It shows graph-average distances,
    # plus the cutoff marker. It should not be forced to start at zero.
    all_mean_distances = [
        float(record["closestInnerOuterDistanceMean"])
        for record in records
        if float(record["closestInnerOuterDistanceMean"]) == float(record["closestInnerOuterDistanceMean"])
    ]
    all_cutoff_values = [
        float(record["distanceSplitCutoff"])
        for record in records
        if float(record["distanceSplitCutoff"]) == float(record["distanceSplitCutoff"])
    ]
    histogram_x_values = all_mean_distances + all_cutoff_values

    if histogram_x_values:
        min_hist_value = min(histogram_x_values)
        max_hist_value = max(histogram_x_values)
        if max_hist_value > min_hist_value:
            histogram_padding = 0.10 * (max_hist_value - min_hist_value)
        else:
            histogram_padding = max(1.0e-6, 0.05 * abs(max_hist_value), 0.05)
        mean_distance_range = [
            min_hist_value - histogram_padding,
            max_hist_value + histogram_padding,
        ]
    else:
        mean_distance_range = [0.0, 1.0]

    distance_bin_count = 80
    distance_bin_size = max(
        1.0e-9,
        (mean_distance_range[1] - mean_distance_range[0]) / float(distance_bin_count),
    )

    all_closest_angle_deg = [
        angle
        for record in records
        for angle in record["closestInnerOuterAngleDeg"]
        if angle == angle
    ]
    if all_closest_angle_deg:
        min_angle = min(all_closest_angle_deg)
        max_angle = max(all_closest_angle_deg)
        if max_angle > min_angle:
            angle_padding = 0.05 * (max_angle - min_angle)
        else:
            angle_padding = max(1.0e-6, 0.05 * abs(max_angle), 0.05)
        angle_range = [
            max(0.0, min_angle - angle_padding),
            max_angle + angle_padding,
        ]
    else:
        angle_range = [0.0, 1.0]

    payload = {
        "viewerVersion": VIEWER_VERSION,
        "graphs": records,
        "axisRange": axis_range,
        "zRange": z_range,
        "distanceRange": distance_range,
        "meanDistanceRange": mean_distance_range,
        "distanceBinSize": distance_bin_size,
        "angleRangeDeg": angle_range,
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
    #class-boundary {{
      font-size: 14px;
    }}
    .current-class-boundary {{
      font-weight: 700;
    }}
    #details {{
      font-size: 13px;
      color: #555555;
    }}
    #viewer-body {{
      display: grid;
      grid-template-columns: minmax(520px, 2fr) minmax(360px, 1fr);
      gap: 12px;
      padding: 12px;
      box-sizing: border-box;
      height: calc(100vh - 118px);
      min-height: 560px;
    }}
    #plot {{
      width: 100%;
      height: 100%;
      min-height: 560px;
      background: white;
      border: 1px solid #e1e1e1;
      border-radius: 10px;
      overflow: hidden;
    }}
    #distance-panel {{
      background: white;
      border: 1px solid #e1e1e1;
      border-radius: 10px;
      padding: 12px;
      min-width: 0;
      overflow-y: auto;
      overflow-x: hidden;
    }}
    #distance-panel h2 {{
      margin: 0 0 6px 0;
      font-size: 16px;
    }}
    .scatter-heading {{
      margin-top: 18px !important;
      padding-top: 6px;
    }}
    #distance-summary {{
      font-size: 12px;
      color: #444444;
      line-height: 1.45;
      margin-bottom: 8px;
    }}
    #distance-histogram {{
      width: 100%;
      height: 260px;
      min-height: 240px;
    }}
    #distance-angle-scatter {{
      width: 100%;
      height: 300px;
      min-height: 280px;
      margin-top: 22px;
      border-top: 1px solid #eeeeee;
      padding-top: 10px;
    }}
    .distance-note {{
      font-size: 12px;
      color: #666666;
      line-height: 1.4;
      margin-top: 8px;
    }}
    @media (max-width: 1100px) {{
      #viewer-body {{
        grid-template-columns: 1fr;
        height: auto;
      }}
      #plot {{
        height: 560px;
      }}
      #distance-panel {{
        overflow: visible;
      }}
      #distance-histogram {{
        height: 260px;
      }}
      #distance-angle-scatter {{
        height: 320px;
      }}
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
      <span id="class-boundary"></span>
      <button id="prev-button" type="button">Previous</button>
      <button id="next-button" type="button">Next</button>
      <button id="play-button" type="button">Play</button>
    </div>
    <div class="control-row">
      <input id="graph-slider" type="range" min="0" max="0" step="1" value="0">
      <span id="status"></span>
    </div>
    <div id="details"></div>
    <div class="hint">The single slider walks through all selected graphs: class 0 first, then class 1. When the slider crosses the class boundary, the status/header changes to show the active class. Rotating or zooming the 3D view is preserved while the slider changes. Viewer template: {VIEWER_VERSION}.</div>
  </div>
  <div id="viewer-body">
    <div id="plot"></div>
    <section id="distance-panel" aria-label="Closest inner-to-outer distance diagnostics">
      <h2>Average closest inner-to-outer distance</h2>
      <div id="distance-summary"></div>
      <div id="distance-histogram"></div>
      <h2 class="scatter-heading">Angle gap vs closest distance</h2>
      <div id="distance-angle-scatter"></div>
      <div class="distance-note">The histogram shows one value per graph: the average nearest outer-ring distance across inner-ring nodes. The dotted vertical line is the class cutoff. The dashed vertical line is the selected graph average. The scatter plot below is unchanged: it shows the raw nearest-node pairs as angle gap vs distance.</div>
    </section>
  </div>

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
      graphsByClass[key].sort((a, b) => {{
        const aIndex = Number.isFinite(Number(a.classGraphIndex)) ? Number(a.classGraphIndex) : Number(a.variationT);
        const bIndex = Number.isFinite(Number(b.classGraphIndex)) ? Number(b.classGraphIndex) : Number(b.variationT);
        return aIndex - bIndex;
      }});
    }}

    const classLabels = Object.keys(graphsByClass).sort((a, b) => Number(a) - Number(b));
    const allGraphs = [];
    for (const label of classLabels) {{
      allGraphs.push(...graphsByClass[label]);
    }}
    let activeClass = allGraphs.length > 0 ? String(allGraphs[0].label) : (classLabels[0] || "0");
    let playTimer = null;
    let storedCamera = null;

    const plotDiv = document.getElementById("plot");
    const histogramDiv = document.getElementById("distance-histogram");
    const angleDistanceDiv = document.getElementById("distance-angle-scatter");
    const distanceSummary = document.getElementById("distance-summary");
    const slider = document.getElementById("graph-slider");
    const status = document.getElementById("status");
    const details = document.getElementById("details");
    const classBoundary = document.getElementById("class-boundary");
    const playButton = document.getElementById("play-button");

    const config = {{responsive: true, displaylogo: false}};
    const classPlotColors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"];
    function colorForClassIndex(index) {{
      return classPlotColors[index % classPlotColors.length];
    }}
    let isProgrammaticCameraUpdate = false;

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

    function attachCameraListener() {{
      if (!plotDiv || plotDiv.__ringCameraListenerAttached || typeof plotDiv.on !== "function") return;
      plotDiv.__ringCameraListenerAttached = true;
      plotDiv.on("plotly_relayout", (eventData) => {{
        if (isProgrammaticCameraUpdate) return;

        if (eventData && eventData["scene.camera"]) {{
          storedCamera = cloneObject(eventData["scene.camera"]);
        }} else {{
          const camera = getCurrentCamera();
          if (camera) storedCamera = camera;
        }}
      }});
    }}

    function currentSliderFraction() {{
      const maxIndex = Math.max(allGraphs.length - 1, 1);
      return Number(slider.value) / maxIndex;
    }}

    function escapeHtml(value) {{
      return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }}

    function updateClassBoundaryText(graph, index) {{
      const boundaryPieces = [];
      let offset = 0;
      for (const label of classLabels) {{
        const graphs = graphsByClass[label] || [];
        if (graphs.length === 0) continue;
        const first = graphs[0];
        const start = offset + 1;
        const end = offset + graphs.length;
        const labelText = `${{first.classDisplay}}: ${{first.className}} ${{start}}-${{end}}`;
        if (String(graph.label) === String(label)) {{
          boundaryPieces.push(`<span class="current-class-boundary">${{escapeHtml(labelText)}} current</span>`);
        }} else {{
          boundaryPieces.push(escapeHtml(labelText));
        }}
        offset += graphs.length;
      }}
      classBoundary.innerHTML = boundaryPieces.join(" | ");
    }}

    function finiteNumberArray(values) {{
      if (!Array.isArray(values)) return [];
      return values.map(Number).filter((value) => Number.isFinite(value));
    }}

    function finiteNumber(value) {{
      const numeric = Number(value);
      return Number.isFinite(numeric) ? numeric : NaN;
    }}

    function formatDistance(value) {{
      return Number.isFinite(value) ? value.toFixed(4) : "n/a";
    }}

    function summarizeDistances(values) {{
      const clean = finiteNumberArray(values);
      if (clean.length === 0) {{
        return {{count: 0, min: NaN, mean: NaN, max: NaN}};
      }}
      let minValue = clean[0];
      let maxValue = clean[0];
      let total = 0.0;
      for (const value of clean) {{
        if (value < minValue) minValue = value;
        if (value > maxValue) maxValue = value;
        total += value;
      }}
      return {{
        count: clean.length,
        min: minValue,
        mean: total / clean.length,
        max: maxValue
      }};
    }}

    function meanDistancesForClass(label) {{
      const graphs = graphsByClass[String(label)] || [];
      const values = [];
      for (const graph of graphs) {{
        const value = finiteNumber(graph.closestInnerOuterDistanceMean);
        if (Number.isFinite(value)) values.push(value);
      }}
      return values;
    }}

    function cutoffForGraph(graph) {{
      const directCutoff = finiteNumber(graph.distanceSplitCutoff);
      if (Number.isFinite(directCutoff)) return directCutoff;

      const low = finiteNumber(graph.distanceSplitThresholdLow);
      const high = finiteNumber(graph.distanceSplitThresholdHigh);
      if (Number.isFinite(low) && Number.isFinite(high)) {{
        return 0.5 * (low + high);
      }}

      return NaN;
    }}

    function cutoffForAllGraphs() {{
      const cutoffs = [];
      for (const graph of allGraphs) {{
        const cutoff = cutoffForGraph(graph);
        if (Number.isFinite(cutoff)) cutoffs.push(cutoff);
      }}
      if (cutoffs.length === 0) return NaN;

      const first = cutoffs[0];
      for (const cutoff of cutoffs) {{
        if (Math.abs(cutoff - first) > 1.0e-9) {{
          // If multiple cutoffs somehow appear, use the mean. This keeps a
          // visible cutoff while avoiding arbitrary first-value behavior.
          let total = 0.0;
          for (const value of cutoffs) total += value;
          return total / cutoffs.length;
        }}
      }}
      return first;
    }}

    function updateDistanceHistogram(graph) {{
      const traces = [];
      const summaryLines = [];
      const classSummaries = [];
      const cutoff = cutoffForAllGraphs();

      for (const label of classLabels) {{
        const graphs = graphsByClass[label] || [];
        if (graphs.length === 0) continue;
        const first = graphs[0];
        const meanDistances = meanDistancesForClass(label);
        const classSummary = summarizeDistances(meanDistances);
        classSummaries.push({{label: label, first: first, distances: meanDistances, summary: classSummary}});
        summaryLines.push(
          `${{first.classDisplay}} (${{first.className}}): graphs=${{classSummary.count}}, min avg=${{formatDistance(classSummary.min)}}, mean avg=${{formatDistance(classSummary.mean)}}, max avg=${{formatDistance(classSummary.max)}}`
        );
      }}

      const shapes = [];
      const annotations = [];

      if (Number.isFinite(cutoff)) {{
        shapes.push({{
          type: "line",
          xref: "x",
          yref: "paper",
          x0: cutoff,
          x1: cutoff,
          y0: 0,
          y1: 1,
          line: {{width: 2, dash: "dot", color: "#666666"}}
        }});
        annotations.push({{
          x: cutoff,
          y: 1.08,
          xref: "x",
          yref: "paper",
          text: "class cutoff",
          showarrow: false,
          font: {{size: 11, color: "#666666"}}
        }});
      }}

      const selectedAverage = finiteNumber(graph.closestInnerOuterDistanceMean);
      if (Number.isFinite(selectedAverage)) {{
        shapes.push({{
          type: "line",
          xref: "x",
          yref: "paper",
          x0: selectedAverage,
          x1: selectedAverage,
          y0: 0,
          y1: 1,
          line: {{width: 3, dash: "dash", color: "#202020"}}
        }});
        annotations.push({{
          x: selectedAverage,
          y: 1.02,
          xref: "x",
          yref: "paper",
          text: "selected graph average",
          showarrow: false,
          font: {{size: 11, color: "#202020"}}
        }});
      }}

      for (let classIndex = 0; classIndex < classSummaries.length; classIndex += 1) {{
        const item = classSummaries[classIndex];
        traces.push({{
          type: "histogram",
          x: item.distances,
          name: `${{item.first.classDisplay}}: ${{item.first.className}}`,
          histnorm: "probability density",
          opacity: 0.88,
          marker: {{color: colorForClassIndex(classIndex), line: {{color: "#202020", width: 1}}}},
          xbins: {{
            start: payload.meanDistanceRange[0],
            end: payload.meanDistanceRange[1],
            size: payload.distanceBinSize
          }},
          hovertemplate: "graph average distance=%{{x:.4f}}<br>density=%{{y:.4f}}<extra>%{{fullData.name}}</extra>"
        }});
      }}

      const selectedRawSummary = summarizeDistances(graph.closestInnerOuterDistances);
      const cutoffLine = Number.isFinite(cutoff)
        ? `<strong>Class cutoff</strong>: average distance = ${{formatDistance(cutoff)}}`
        : `<strong>Class cutoff</strong>: n/a`;
      const selectedLine = `<strong>Selected graph average</strong>: ${{formatDistance(selectedAverage)}}`;
      const rawLine = `<strong>Selected raw nearest distances</strong>: n=${{selectedRawSummary.count}}, min=${{formatDistance(selectedRawSummary.min)}}, mean=${{formatDistance(selectedRawSummary.mean)}}, max=${{formatDistance(selectedRawSummary.max)}}`;

      distanceSummary.innerHTML = [
        selectedLine,
        cutoffLine,
        rawLine,
        ...summaryLines
      ].filter((line) => line.length > 0).join("<br>");

      const layout = {{
        xaxis: {{
          title: "average closest distance from inner nodes to outer ring",
          range: payload.meanDistanceRange
        }},
        yaxis: {{title: "probability density", rangemode: "tozero"}},
        barmode: "overlay",
        bargap: 0.08,
        bargroupgap: 0.02,
        margin: {{l: 56, r: 16, t: 72, b: 56}},
        legend: {{orientation: "h", y: 1.18, x: 0.0}},
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        font: {{color: "#202020"}},
        shapes: shapes,
        annotations: annotations
      }};
      Plotly.react(histogramDiv, traces, layout, config);
    }}

    function updateDistanceAngleScatter(graph) {{
      const traces = [];
      for (let classIndex = 0; classIndex < classLabels.length; classIndex += 1) {{
        const label = classLabels[classIndex];
        const graphs = graphsByClass[label] || [];
        if (graphs.length === 0) continue;
        const first = graphs[0];
        const xAngles = [];
        const yDistances = [];
        const hoverText = [];
        for (const classGraph of graphs) {{
          const angles = finiteNumberArray(classGraph.closestInnerOuterAngleDeg);
          const distances = finiteNumberArray(classGraph.closestInnerOuterDistances);
          const n = Math.min(angles.length, distances.length);
          for (let i = 0; i < n; i += 1) {{
            xAngles.push(angles[i]);
            yDistances.push(distances[i]);
            hoverText.push(`${{classGraph.classDisplay}} (${{classGraph.className}})<br>graph=${{classGraph.name}}<br>angle gap=${{angles[i].toFixed(3)}} deg<br>closest distance=${{distances[i].toFixed(4)}}`);
          }}
        }}
        traces.push({{
          type: "scatter",
          mode: "markers",
          x: xAngles,
          y: yDistances,
          name: `${{first.classDisplay}}: ${{first.className}}`,
          marker: {{
            color: colorForClassIndex(classIndex),
            size: 10,
            opacity: 0.85,
            line: {{color: "#202020", width: 1}}
          }},
          text: hoverText,
          hoverinfo: "text"
        }});
      }}

      const selectedAngles = finiteNumberArray(graph.closestInnerOuterAngleDeg);
      const selectedDistances = finiteNumberArray(graph.closestInnerOuterDistances);
      const selectedN = Math.min(selectedAngles.length, selectedDistances.length);
      if (selectedN > 0) {{
        traces.push({{
          type: "scatter",
          mode: "markers",
          x: selectedAngles.slice(0, selectedN),
          y: selectedDistances.slice(0, selectedN),
          name: "selected graph",
          marker: {{
            color: "#000000",
            size: 15,
            symbol: "x",
            line: {{color: "#000000", width: 2}}
          }},
          hovertemplate: "selected graph<br>angle=%{{x:.3f}} deg<br>distance=%{{y:.4f}}<extra></extra>"
        }});
      }}

      const layout = {{
        title: {{text: "Angle gap vs closest distance", x: 0.02, xanchor: "left"}},
        xaxis: {{
          title: "nearest inner-to-outer angle gap (degrees)",
          range: payload.angleRangeDeg,
          gridcolor: "#e5e5e5",
          zerolinecolor: "#999999"
        }},
        yaxis: {{
          title: "closest distance",
          range: payload.distanceRange,
          gridcolor: "#e5e5e5",
          zerolinecolor: "#999999"
        }},
        margin: {{l: 56, r: 16, t: 52, b: 56}},
        legend: {{orientation: "h", y: 1.18, x: 0.0}},
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        font: {{color: "#202020"}}
      }};
      Plotly.react(angleDistanceDiv, traces, layout, config);
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
        uirevision: "keep-camera",
        margin: {{l: 0, r: 0, t: 78, b: 0}},
        showlegend: false
      }};
      if (camera) {{
        layout.scene.camera = cloneObject(camera);
      }}
      return layout;
    }}

    function updatePlot() {{
      if (allGraphs.length === 0) return;
      const maxIndex = Math.max(allGraphs.length - 1, 0);
      slider.max = String(maxIndex);
      let index = Number(slider.value);
      if (!Number.isFinite(index)) index = 0;
      index = Math.max(0, Math.min(maxIndex, Math.round(index)));
      slider.value = String(index);

      const graph = allGraphs[index];
      activeClass = String(graph.label);
      const camera = storedCamera ? cloneObject(storedCamera) : getCurrentCamera();
      Plotly.react(plotDiv, makeTraces(graph), makeLayout(graph, camera), config).then(() => {{
        attachCameraListener();
        if (camera) {{
          storedCamera = cloneObject(camera);
          isProgrammaticCameraUpdate = true;
          const relayoutPromise = Plotly.relayout(plotDiv, {{"scene.camera": cloneObject(camera)}});
          Promise.resolve(relayoutPromise).then(() => {{
            isProgrammaticCameraUpdate = false;
          }}, () => {{
            isProgrammaticCameraUpdate = false;
          }});
        }}
      }});
      status.textContent = `${{graph.classDisplay}}: ${{graph.className}} | graph ${{index + 1}}/${{allGraphs.length}} | class graph ${{graph.classGraphIndex + 1}}/${{graph.classGraphCount}} | variation ${{(100 * graph.variationT).toFixed(1)}}%`;
      details.textContent = `dataset index ${{graph.originalDatasetIndex}} | ${{graph.name}} | inner radius ${{graph.innerRadius.toFixed(3)}} | outer radius ${{graph.outerRadius.toFixed(3)}} | outer rotation ${{graph.outerRotationDeg.toFixed(1)}} deg | outer phase ${{graph.outerPhaseDeg.toFixed(1)}} deg | outer 3D tilt ${{graph.outer3DRotationDeg.toFixed(1)}} deg | tilt axis ${{graph.outer3DAxisDeg.toFixed(1)}} deg | closest inner-outer mean ${{formatDistance(graph.closestInnerOuterDistanceMean)}}`;
      updateClassBoundaryText(graph, index);
      updateDistanceHistogram(graph);
      updateDistanceAngleScatter(graph);
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
        const maxIndex = Math.max(allGraphs.length - 1, 0);
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

    slider.max = String(Math.max(allGraphs.length - 1, 0));
    updatePlot();
  </script>
</body>
</html>
"""

    _validate_generated_viewer_html(html)
    html_path.write_text(html, encoding="utf-8")